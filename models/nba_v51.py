"""v51: Denoising Diffusion Probabilistic Model (DDPM) for trajectory prediction.
Context encoder (same GRU as v44) produces h_context per player.
Denoiser: MLP(noisy_future_flat, sinusoidal_t_emb, h_context) → predicted noise.
Training: randomly sample diffusion step t, add noise, predict noise, MSE loss.
Inference: reverse chain from pure noise x_T to x_0, return x_0 reshaped as trajectory.
T_DIFF=100, 150 epochs (more epochs needed for diffusion to converge).
Reference: LED (NeurIPS 2022), MID (CVPR 2022).
"""
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

_DIR = Path(__file__).parent
WEIGHTS_DIR = _DIR / 'weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TXT = _DIR.parent / 'train_files.txt'
VAL_TXT   = _DIR.parent / 'val_files.txt'

TEAM_EMB_DIM = 8
T_DIFF = 100       # diffusion steps
HORIZON = 12
CONTEXT = 8
N_AGENTS = 11


def cosine_beta_schedule(T, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal 2021) — less noise early."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0, 0.999).float()


class NBADataset(Dataset):
    def __init__(self, file_list, context_size, horizon_size, mu, sigma, augment=False):
        super().__init__()
        self.context_size = context_size; self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size; self.augment = augment
        self.sequences = []; self.max_start = []
        for f in file_list:
            seq = torch.load(f, weights_only=False)
            seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - mu) / sigma
            self.sequences.append(seq)
            self.max_start.append(max(0, len(seq) - self.window_size))

    def __getitem__(self, index):
        seq_idx, start = index
        X = self.sequences[seq_idx][start:start + self.context_size].clone()
        y = self.sequences[seq_idx][start + self.context_size:start + self.window_size].clone()
        if self.augment:
            if random.random() < 0.5:
                X[:, :, 0] = -X[:, :, 0]; y[:, :, 0] = -y[:, :, 0]
            if random.random() < 0.5:
                X[:, :, 1] = -X[:, :, 1]; y[:, :, 1] = -y[:, :, 1]
        return X, y

    def __len__(self):
        return len(self.sequences)


class NBASampler(Sampler):
    def __init__(self, batch_size, max_start, seed=0, shuffle=True):
        self.batch_size = batch_size; self.max_start = max_start
        self.epoch = 0; self.seed = seed; self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def set_epoch(self, epoch):
        self.epoch = epoch; self.generator.manual_seed(self.seed + epoch)

    def __iter__(self):
        n = len(self)
        perm = torch.randperm(n, generator=self.generator).tolist() if self.shuffle else list(range(n))
        perm_start = [(i, self.max_start[i]) for i in perm]
        for k in range(0, n, self.batch_size):
            for idx, ms in perm_start[k:k + self.batch_size]:
                yield idx, torch.randint(0, ms + 1, size=(), generator=self.generator)

    def __len__(self):
        return len(self.max_start)


class GRUCell(nn.Module):
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim)
        self.GRU_Z = nn.Sequential(nn.Linear(input_dim + state_dim, state_dim), nn.Sigmoid())
        self.GRU_R = nn.Sequential(nn.Linear(input_dim + state_dim, state_dim), nn.Sigmoid())
        self.GRU_H = nn.Sequential(nn.Linear(input_dim + state_dim, state_dim), nn.Tanh())

    def forward(self, x, h):
        x = self.pre_norm(x)
        Z = self.GRU_Z(torch.cat([x, h], dim=1))
        R = self.GRU_R(torch.cat([x, h], dim=1))
        return Z * h + (1 - Z) * self.GRU_H(torch.cat([x, R * h], dim=1))


class TypeAwareDistanceBiasedAttention(nn.Module):
    def __init__(self, state_dim, num_heads=4, num_types=3):
        super().__init__()
        self.num_heads = num_heads; self.head_dim = state_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(state_dim, state_dim); self.k = nn.Linear(state_dim, state_dim)
        self.v = nn.Linear(state_dim, state_dim); self.out = nn.Linear(state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)
        self.dist_proj = nn.Linear(1, num_heads)
        self.type_pair_bias = nn.Embedding(num_types * num_types, num_heads)

    def forward(self, h, pos, node_type):
        B, N, D = h.shape
        dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1, keepdim=True)
        dist_bias = self.dist_proj(dist).permute(0, 3, 1, 2)
        type_i = node_type.unsqueeze(2).expand(-1, -1, N)
        type_j = node_type.unsqueeze(1).expand(-1, N, -1)
        type_bias = self.type_pair_bias(type_i * 3 + type_j).permute(0, 3, 1, 2)
        def split(t): return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        Q, K, V = split(self.q(h)), split(self.k(h)), split(self.v(h))
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale + dist_bias + type_bias, dim=-1)
        return self.norm(h + self.out((attn @ V).transpose(1, 2).reshape(B, N, D)))


def sinusoidal_embedding(t, dim):
    """t: [N] integer steps → [N, dim] sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionDenoiser(nn.Module):
    """MLP denoiser: (noisy_traj_flat, t_emb, h_context) → predicted_noise_flat."""
    def __init__(self, traj_dim, context_dim, t_emb_dim=64, hidden=512):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.t_proj = nn.Sequential(nn.Linear(t_emb_dim, t_emb_dim * 2), nn.SiLU(), nn.Linear(t_emb_dim * 2, t_emb_dim))
        in_dim = traj_dim + t_emb_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, traj_dim),
        )

    def forward(self, x_noisy, t, h_ctx):
        """x_noisy: [BN, traj_dim], t: [BN] int, h_ctx: [BN, ctx_dim]."""
        t_emb = sinusoidal_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        inp = torch.cat([x_noisy, t_emb, h_ctx], dim=-1)
        return self.net(inp)


class NBAv51Model(nn.Module):
    """DDPM: diffusion over flat trajectory, conditioned on GRU context."""

    def __init__(self, state_dim=128, context_size=CONTEXT, horizon_size=HORIZON, num_heads=4, num_layers=2):
        super().__init__()
        self.context_size = context_size; self.horizon_size = horizon_size
        self.state_dim = state_dim; self.num_layers = num_layers

        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        input_dim_0 = 32 + TEAM_EMB_DIM
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.gru_layers = nn.ModuleList([GRUCell(input_dim_0 if i == 0 else state_dim, state_dim) for i in range(num_layers)])
        self.attn_layers = nn.ModuleList([TypeAwareDistanceBiasedAttention(state_dim, num_heads) for _ in range(num_layers)])

        # Trajectory dim: horizon * 2 (x,y per step, per player — denoised per-player)
        traj_dim = horizon_size * 2
        self.denoiser = DiffusionDenoiser(traj_dim=traj_dim, context_dim=state_dim)

        # Pre-compute diffusion schedule
        betas = cosine_beta_schedule(T_DIFF)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1 - alphas_cumprod).sqrt())
        self.register_buffer('sqrt_recip_alphas', (1.0 / alphas).sqrt())
        posterior_var = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer('posterior_var', posterior_var.clamp(min=1e-20))

    def _get_node_type(self, X):
        team = X[:, 0, :, 3]; node_type = torch.zeros_like(team, dtype=torch.long)
        node_type[team == -1] = 1; node_type[team == 1] = 2; return node_type

    def _ball_broadcast(self, pos, vel, ball_mask):
        m = ball_mask.float().unsqueeze(-1)
        return (pos * m).sum(1, keepdim=True).expand_as(pos), (vel * m).sum(1, keepdim=True).expand_as(vel)

    def _social_features(self, pos, vel, node_type, ball_mask):
        B, N, _ = pos.shape; INF = 1e4
        dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)
        eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
        type_i = node_type.unsqueeze(2); type_j = node_type.unsqueeze(1)
        same_mask = (type_i == type_j) & (type_i != 0) & ~eye
        opp_mask = ((type_i == 1) & (type_j == 2)) | ((type_i == 2) & (type_j == 1))
        dist_same = dist.masked_fill(~same_mask, INF); dist_opp = dist.masked_fill(~opp_mask, INF)
        nn_s = dist_same.topk(2, dim=2, largest=False).indices; nn_o = dist_opp.topk(2, dim=2, largest=False).indices
        b_idx = torch.arange(B, device=pos.device).unsqueeze(1).expand(B, N)
        ball2 = ball_mask.unsqueeze(-1).expand(B, N, 2)
        def gr(idx): return (pos[b_idx, idx] - pos).masked_fill(ball2, 0.), (vel[b_idx, idx] - vel).masked_fill(ball2, 0.)
        rp_s1, rv_s1 = gr(nn_s[:,:,0]); rp_s2, rv_s2 = gr(nn_s[:,:,1])
        rp_o1, rv_o1 = gr(nn_o[:,:,0]); rp_o2, rv_o2 = gr(nn_o[:,:,1])
        T = self.log_temperature.exp().clamp(0.1, 10.)
        w_s = torch.softmax(-dist_same / T, dim=2); w_o = torch.softmax(-dist_opp / T, dim=2)
        rp = pos.unsqueeze(1) - pos.unsqueeze(2); rv = vel.unsqueeze(1) - vel.unsqueeze(2)
        aps = (w_s.unsqueeze(-1) * rp).sum(2).masked_fill(ball2, 0.); avs = (w_s.unsqueeze(-1) * rv).sum(2).masked_fill(ball2, 0.)
        apo = (w_o.unsqueeze(-1) * rp).sum(2).masked_fill(ball2, 0.); avo = (w_o.unsqueeze(-1) * rv).sum(2).masked_fill(ball2, 0.)
        return torch.cat([rp_s1, rv_s1, rp_s2, rv_s2, rp_o1, rv_o1, rp_o2, rv_o2, aps, avs, apo, avo], dim=-1)

    def _encode_context(self, X):
        B, _, N, _ = X.shape
        ball_mask = (X[:, 0, :, 2] == 0); node_type = self._get_node_type(X)
        team_emb = self.team_emb(node_type)
        h = [torch.zeros(B * N, self.state_dim, device=X.device) for _ in range(self.num_layers)]
        prev_pos = None
        for t in range(self.context_size):
            pos = X[:, t, :, :2]
            vel = torch.zeros_like(pos) if prev_pos is None else pos - prev_pos
            prev_pos = pos
            ball_pos, ball_vel = self._ball_broadcast(pos, vel, ball_mask)
            motion = torch.cat([pos, vel, pos - ball_pos, vel - ball_vel], dim=-1)
            social = self._social_features(pos, vel, node_type, ball_mask)
            x_in = torch.cat([motion, social, team_emb], dim=-1).reshape(B * N, -1)
            layer_in = x_in
            for i in range(self.num_layers):
                h[i] = self.gru_layers[i](layer_in, h[i])
                h[i] = self.attn_layers[i](h[i].reshape(B, N, self.state_dim), pos, node_type).reshape(B * N, self.state_dim)
                layer_in = h[i]
        return h[-1], ball_mask, node_type  # [B*N, state_dim]

    def q_sample(self, x0, t, noise):
        """Forward diffusion: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise."""
        sqrt_ab = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def forward(self, X, y=None):
        B, _, N, _ = X.shape
        BN = B * N
        h_ctx, ball_mask, node_type = self._encode_context(X)  # [B*N, D]

        if self.training and y is not None:
            # y: [B, T, N, F] → flatten future positions to [B*N, T*2]
            y_pos = y[:, :, :, :2].permute(0, 2, 1, 3).reshape(BN, self.horizon_size * 2)
            t_steps = torch.randint(0, T_DIFF, (BN,), device=X.device)
            noise = torch.randn_like(y_pos)
            y_noisy = self.q_sample(y_pos, t_steps, noise)
            noise_pred = self.denoiser(y_noisy, t_steps, h_ctx)
            return F.mse_loss(noise_pred, noise)

        # Inference: reverse diffusion chain
        x = torch.randn(BN, self.horizon_size * 2, device=X.device)
        for t_val in reversed(range(T_DIFF)):
            t_tensor = torch.full((BN,), t_val, device=X.device, dtype=torch.long)
            noise_pred = self.denoiser(x, t_tensor, h_ctx)
            # DDPM reverse step
            beta_t = self.betas[t_val]
            sqrt_recip = self.sqrt_recip_alphas[t_val]
            sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t_val]
            x0_hat = sqrt_recip * (x - beta_t / sqrt_1mab * noise_pred)
            if t_val > 0:
                noise = torch.randn_like(x)
                x = x0_hat + self.posterior_var[t_val].sqrt() * noise
            else:
                x = x0_hat
        # x: [B*N, T*2] → [T, B*N, 2]
        return x.reshape(B * N, self.horizon_size, 2).permute(1, 0, 2)


class DiffusionLossWrapper:
    """Wraps the model's own MSE loss (model returns scalar loss during training)."""
    def compute(self, loss_or_pred, target):
        if isinstance(loss_or_pred, torch.Tensor) and loss_or_pred.dim() == 0:
            return loss_or_pred  # scalar MSE from model
        # fallback MSE (shouldn't happen in normal flow)
        B, T, N, _ = target.shape
        tgt = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        return sum(F.mse_loss(loss_or_pred[t], tgt[t]) for t in range(T)) / T


class MultiStepMSE:
    def __init__(self): self.loss_fn = nn.MSELoss()

    def compute(self, pred, target):
        B, T, N, F = target.shape
        target = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        return sum(self.loss_fn(pred[t], target[t]) for t in range(T)) / T


class Trainer:
    def __init__(self, train_path, val_path, device='cuda', epochs=35, batch_size=32, **kwargs):
        self.device = device; self.epochs = epochs; self.batch_size = batch_size; self.model = None
        train_files = TRAIN_TXT.read_text().splitlines() if TRAIN_TXT.exists() else [str(p) for p in Path(train_path).glob('*.pt')]
        val_files   = VAL_TXT.read_text().splitlines()   if VAL_TXT.exists()   else [str(p) for p in Path(val_path).glob('*.pt')]
        mu, sigma = self._compute_normalization(train_files)
        self.mu = mu; self.sigma = sigma
        self.train_dataset = NBADataset(train_files, CONTEXT, HORIZON, mu, sigma, augment=True)
        self.val_dataset   = NBADataset(val_files,   CONTEXT, HORIZON, mu, sigma, augment=False)

    def _compute_normalization(self, file_list):
        all_pos = [torch.load(f, weights_only=False)[:, :, [0, 1]] for f in file_list]
        all_pos = torch.cat(all_pos, dim=0)
        return all_pos.mean(dim=(0, 1)), all_pos.std(dim=(0, 1))

    def train(self):
        torch.manual_seed(0); random.seed(0); generator = torch.Generator('cpu').manual_seed(0)
        train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start)
        val_sampler   = NBASampler(self.batch_size, self.val_dataset.max_start, shuffle=False)
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, generator=generator)
        val_dl   = DataLoader(self.val_dataset,   batch_size=self.batch_size, sampler=val_sampler,   generator=generator)
        model = NBAv51Model().to(self.device)
        train_loss_fn = DiffusionLossWrapper(); val_loss_fn = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            model.train(); self._run_epoch(train_dl, train_loss_fn, model, optimizer, 'train')
            model.train(False)
            with torch.no_grad(): self._run_epoch(val_dl, val_loss_fn, model, None, 'val')
            train_sampler.set_epoch(epoch); val_sampler.set_epoch(epoch); scheduler.step()
        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v51.pth')
        self.model = model

    def _run_epoch(self, dl, loss_fn, model, opt, split):
        avg = 0.0
        for X, y in tqdm(dl, desc=f'  {split}'):
            X, y = X.to(self.device), y.to(self.device)
            pred = model(X, y) if split == 'train' else model(X)
            loss = loss_fn.compute(pred, y)
            if split == 'train':
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
            avg += loss.detach()
        print(f'    {split} loss: {(avg / len(dl)).item():.4f}')
