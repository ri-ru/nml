"""v57: v50 (Space-Time Transformer encoder + GRU decoder + cross-attn) + MoG K=2 head.
The "missing" cross-architecture experiment: v50 used Gaussian NLL,
v46 used MoG. v57 = v50 inductive bias + v46 output distribution.
Goal: ensemble decorrelation against v46 — different inductive bias, same loss.
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
D_MODEL = 128
N_HEADS = 4
N_LAYERS_ENC = 3
N_LAYERS_DEC = 2
CONTEXT = 8
HORIZON = 12
N_AGENTS = 11
K_MIX = 2


class NBADataset(Dataset):
    def __init__(self, file_list, context_size, horizon_size, mu, sigma, augment=False):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.augment = augment
        self.sequences = []
        self.max_start = []
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


class MoGNLLLoss:
    """Timestep-weighted MoG NLL with K=2 components."""
    def compute(self, pred, target):
        B, T, N, _ = target.shape
        tgt = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        pred = pred.reshape(T, B * N, K_MIX, 5)
        mu        = pred[:, :, :, :2]
        log_sigma = pred[:, :, :, 2:4].clamp(-3, 3)
        logit_w   = pred[:, :, :, 4]
        log_w = logit_w - logit_w.logsumexp(dim=-1, keepdim=True)
        var = (log_sigma * 2).exp()
        tgt_k = tgt.unsqueeze(2).expand_as(mu)
        diff = tgt_k - mu
        log_prob_k = -0.5 * (diff ** 2 / var + log_sigma * 2 + math.log(2 * math.pi))
        log_prob_k = log_prob_k.sum(dim=-1)
        log_mix = torch.logsumexp(log_w + log_prob_k, dim=-1)
        weights = torch.linspace(1.0, 2.0, T, device=pred.device)
        total = sum(weights[t] * (-log_mix[t].mean()) for t in range(T))
        return total / weights.sum()


class MultiStepMSE:
    def __init__(self): self.loss_fn = nn.MSELoss()

    def compute(self, pred, target):
        B, T, N, F = target.shape
        target = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        return sum(self.loss_fn(pred[t], target[t]) for t in range(T)) / T


class SpaceTimeTransformerEncoder(nn.Module):
    """Encodes [B, T_ctx, N, feat_dim] → [B, N, D_MODEL] per-player summary."""

    def __init__(self, feat_dim, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS_ENC, n_types=3):
        super().__init__()
        self.d_model = d_model
        self.feat_proj = nn.Linear(feat_dim, d_model)
        # Separate embeddings for time and agent type
        self.time_emb = nn.Embedding(CONTEXT, d_model)
        self.type_emb = nn.Embedding(n_types, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Pool over time: learnable query per agent position
        self.pool = nn.Linear(d_model, d_model)

    def forward(self, tokens, node_type):
        """
        tokens: [B, T, N, feat_dim]
        node_type: [B, N] — 0=ball, 1=team_a, 2=team_b
        Returns: [B, N, d_model]
        """
        B, T, N, _ = tokens.shape
        x = self.feat_proj(tokens)  # [B, T, N, D]

        t_idx = torch.arange(T, device=tokens.device).unsqueeze(1)  # [T, 1]
        x = x + self.time_emb(t_idx).unsqueeze(0)  # broadcast over B, N

        type_e = self.type_emb(node_type).unsqueeze(1)  # [B, 1, N, D]
        x = x + type_e  # [B, T, N, D]

        # Flatten (T, N) → single sequence of T*N tokens
        x_flat = x.reshape(B, T * N, self.d_model)  # [B, T*N, D]
        x_enc = self.transformer(x_flat)              # [B, T*N, D]

        # Re-shape and pool over time → per-agent summary
        x_enc = x_enc.reshape(B, T, N, self.d_model)  # [B, T, N, D]
        summary = self.pool(x_enc[:, -1, :, :])        # use last timestep [B, N, D]
        return summary


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


class NBAv57Model(nn.Module):
    """Space-Time Transformer encoder + GRU decoder + MoG K=2 output head."""

    def __init__(self, state_dim=D_MODEL, context_size=CONTEXT, horizon_size=HORIZON):
        super().__init__()
        self.context_size = context_size; self.horizon_size = horizon_size
        self.state_dim = state_dim

        # Input features: pos(2) + vel(2) + ball_rel_pos(2) + ball_rel_vel(2) = 8
        self.encoder = SpaceTimeTransformerEncoder(feat_dim=8, d_model=state_dim)

        # Decoder GRU: takes local motion + social summary (from cross-attn) + team emb
        # Input: pos(2) + vel(2) + ball_rel(4) + team_emb(TEAM_EMB_DIM)
        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        dec_input_dim = 8 + TEAM_EMB_DIM
        self.dec_gru = GRUCell(dec_input_dim, state_dim)

        # Cross-attention: decoder query attends to encoder keys (from transformer summary)
        self.cross_attn = nn.MultiheadAttention(state_dim, N_HEADS, batch_first=True, dropout=0.0)
        self.cross_norm = nn.LayerNorm(state_dim)

        self.proj = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, K_MIX * 5),
        )
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def _get_node_type(self, X):
        team = X[:, 0, :, 3]; node_type = torch.zeros_like(team, dtype=torch.long)
        node_type[team == -1] = 1; node_type[team == 1] = 2; return node_type

    def _ball_broadcast(self, pos, vel, ball_mask):
        m = ball_mask.float().unsqueeze(-1)
        return (pos * m).sum(1, keepdim=True).expand_as(pos), (vel * m).sum(1, keepdim=True).expand_as(vel)

    def _build_context_tokens(self, X):
        """Build per-(time, agent) feature tokens from context window."""
        B, T, N, _ = X.shape
        ball_mask = (X[:, 0, :, 2] == 0)
        tokens = []
        prev_pos = None
        for t in range(T):
            pos = X[:, t, :, :2]
            vel = torch.zeros_like(pos) if prev_pos is None else pos - prev_pos
            prev_pos = pos
            ball_pos, ball_vel = self._ball_broadcast(pos, vel, ball_mask)
            feat = torch.cat([pos, vel, pos - ball_pos, vel - ball_vel], dim=-1)  # [B, N, 8]
            tokens.append(feat)
        return torch.stack(tokens, dim=1)  # [B, T, N, 8]

    def forward(self, X):
        B, _, N, _ = X.shape
        ball_mask = (X[:, 0, :, 2] == 0); node_type = self._get_node_type(X)
        team_emb = self.team_emb(node_type)  # [B, N, TEAM_EMB_DIM]

        # Encode full context with Space-Time Transformer
        ctx_tokens = self._build_context_tokens(X)   # [B, T, N, 8]
        enc_out = self.encoder(ctx_tokens, node_type)  # [B, N, D]

        # Initialize decoder hidden state from encoder summary
        h = enc_out.reshape(B * N, self.state_dim)

        all_preds = []; prev_pos = X[:, -1, :, :2]; pred_mu = None

        for t in range(self.horizon_size):
            pos = prev_pos if t == 0 else pred_mu.reshape(B, N, 2)
            vel = pos - prev_pos if t > 0 else torch.zeros_like(pos)
            prev_pos = pos

            ball_pos, ball_vel = self._ball_broadcast(pos, vel, ball_mask)
            motion = torch.cat([pos, vel, pos - ball_pos, vel - ball_vel], dim=-1)
            x_in = torch.cat([motion, team_emb], dim=-1).reshape(B * N, -1)

            h = self.dec_gru(x_in, h)

            # Cross-attention: each agent's h queries the full encoder output
            h_q = h.reshape(B, N, self.state_dim)
            h_attn, _ = self.cross_attn(h_q, enc_out, enc_out)
            h = self.cross_norm(h_q + h_attn).reshape(B * N, self.state_dim)

            out = self.proj(h)  # [B*N, K*5]
            out_k = out.reshape(B * N, K_MIX, 5)
            log_w = out_k[:, :, 4] - out_k[:, :, 4].logsumexp(dim=-1, keepdim=True)
            w = log_w.exp()
            pred_mu = (w.unsqueeze(-1) * out_k[:, :, :2]).sum(dim=1)
            all_preds.append(out)

        preds = torch.stack(all_preds, dim=0)  # [T, B*N, K*5]
        if self.training:
            return preds
        preds_k = preds.reshape(preds.shape[0], B * N, K_MIX, 5)
        log_w = preds_k[:, :, :, 4] - preds_k[:, :, :, 4].logsumexp(dim=-1, keepdim=True)
        w = log_w.exp()
        return (w.unsqueeze(-1) * preds_k[:, :, :, :2]).sum(dim=2)


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
        model = NBAv57Model().to(self.device)
        train_loss_fn = MoGNLLLoss(); val_loss_fn = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            model.train(); self._run_epoch(train_dl, train_loss_fn, model, optimizer, 'train')
            model.train(False)
            with torch.no_grad(): self._run_epoch(val_dl, val_loss_fn, model, None, 'val')
            train_sampler.set_epoch(epoch); val_sampler.set_epoch(epoch); scheduler.step()
        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v57.pth')
        self.model = model

    def _run_epoch(self, dl, loss_fn, model, opt, split):
        avg = 0.0
        for X, y in tqdm(dl, desc=f'  {split}'):
            X, y = X.to(self.device), y.to(self.device)
            pred = model(X)
            loss = loss_fn.compute(pred, y)
            if split == 'train':
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
            avg += loss.detach()
        print(f'    {split} loss: {(avg / len(dl)).item():.4f}')
