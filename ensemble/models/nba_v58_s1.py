"""v58_s1: v46 architecture trained on train_files.txt + v46×300ep TTA pseudo-labels for test set.
Self-distillation / pseudo-labeling: all 1243 test samples become pseudo-labeled training samples
where the "ground truth" future is the v46 TTA prediction. Risk: amplifies model bias.
Mitigation: pseudo samples make up ~24% of training data, real labels still dominate.
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

TRAIN_TXT_PSEUDO = _DIR.parent / 'train_files_pseudo.txt'
TRAIN_TXT = _DIR.parent / 'train_files.txt'
VAL_TXT   = _DIR.parent / 'val_files.txt'

TEAM_EMB_DIM = 8
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
                X[:, :, 0] = -X[:, :, 0]
                y[:, :, 0] = -y[:, :, 0]
            if random.random() < 0.5:
                X[:, :, 1] = -X[:, :, 1]
                y[:, :, 1] = -y[:, :, 1]
        return X, y

    def __len__(self):
        return len(self.sequences)


class NBASampler(Sampler):
    def __init__(self, batch_size, max_start, seed=0, shuffle=True):
        self.batch_size = batch_size
        self.max_start = max_start
        self.epoch = 0
        self.generator = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.shuffle = shuffle

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.generator.manual_seed(self.seed + epoch)

    def __iter__(self):
        n = len(self)
        perm = torch.randperm(n, generator=self.generator).tolist() if self.shuffle else list(range(n))
        perm_start = [(i, self.max_start[i]) for i in perm]
        for k in range(0, n, self.batch_size):
            for idx, max_start in perm_start[k:k + self.batch_size]:
                start = torch.randint(0, max_start + 1, size=(), generator=self.generator)
                yield idx, start

    def __len__(self):
        return len(self.max_start)


class MoGNLLLoss:
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
    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def compute(self, pred, target):
        B, T, N, F = target.shape
        target = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        return sum(self.loss_fn(pred[t], target[t]) for t in range(T)) / T


class GRUCell(nn.Module):
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.pre_norm = nn.LayerNorm(input_dim)
        self.GRU_Z = nn.Sequential(nn.Linear(input_dim + state_dim, state_dim), nn.Sigmoid())
        self.GRU_R = nn.Sequential(nn.Linear(input_dim + state_dim, state_dim), nn.Sigmoid())
        self.GRU_H = nn.Sequential(nn.Linear(input_dim + state_dim, state_dim), nn.Tanh())

    def forward(self, x, h):
        x = self.pre_norm(x)
        Z = self.GRU_Z(torch.cat([x, h], dim=1))
        R = self.GRU_R(torch.cat([x, h], dim=1))
        H_tilde = self.GRU_H(torch.cat([x, R * h], dim=1))
        return Z * h + (1 - Z) * H_tilde


class TypeAwareDistanceBiasedAttention(nn.Module):
    def __init__(self, state_dim, num_heads=4, num_types=3):
        super().__init__()
        assert state_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = state_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(state_dim, state_dim)
        self.k = nn.Linear(state_dim, state_dim)
        self.v = nn.Linear(state_dim, state_dim)
        self.out = nn.Linear(state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)
        self.dist_proj = nn.Linear(1, num_heads)
        self.type_pair_bias = nn.Embedding(num_types * num_types, num_heads)

    def forward(self, h, pos, node_type):
        B, N, D = h.shape
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = diff.norm(dim=-1, keepdim=True)
        dist_bias = self.dist_proj(dist).permute(0, 3, 1, 2)
        type_i = node_type.unsqueeze(2).expand(-1, -1, N)
        type_j = node_type.unsqueeze(1).expand(-1, N, -1)
        type_bias = self.type_pair_bias(type_i * 3 + type_j).permute(0, 3, 1, 2)

        def split_heads(t):
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(self.q(h)), split_heads(self.k(h)), split_heads(self.v(h))
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale + dist_bias + type_bias, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.norm(h + self.out(out))


class NBAv58Model(nn.Module):
    """v46 arch retrained on real + pseudo-labeled test data."""

    def __init__(self, state_dim=128, context_size=8, horizon_size=12, num_heads=4, num_layers=2):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.state_dim = state_dim
        self.num_layers = num_layers

        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        input_dim_0 = 32 + TEAM_EMB_DIM

        self.log_temperature = nn.Parameter(torch.zeros(1))

        self.gru_layers = nn.ModuleList([
            GRUCell(input_dim_0 if i == 0 else state_dim, state_dim)
            for i in range(num_layers)
        ])
        self.attn_layers = nn.ModuleList([
            TypeAwareDistanceBiasedAttention(state_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.proj = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, K_MIX * 5),
        )

    def _get_node_type(self, X):
        team = X[:, 0, :, 3]
        node_type = torch.zeros_like(team, dtype=torch.long)
        node_type[team == -1] = 1
        node_type[team == 1] = 2
        return node_type

    def _ball_broadcast(self, pos, vel, ball_mask):
        mask_f = ball_mask.float().unsqueeze(-1)
        ball_pos = (pos * mask_f).sum(dim=1, keepdim=True).expand_as(pos)
        ball_vel = (vel * mask_f).sum(dim=1, keepdim=True).expand_as(vel)
        return ball_pos, ball_vel

    def _social_features(self, pos, vel, node_type, ball_mask):
        B, N, _ = pos.shape
        INF = 1e4
        dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)
        eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
        type_i = node_type.unsqueeze(2)
        type_j = node_type.unsqueeze(1)
        same_mask = (type_i == type_j) & (type_i != 0) & ~eye
        opp_mask = ((type_i == 1) & (type_j == 2)) | ((type_i == 2) & (type_j == 1))
        dist_same = dist.masked_fill(~same_mask, INF)
        dist_opp  = dist.masked_fill(~opp_mask,  INF)
        nn_same_top2 = dist_same.topk(2, dim=2, largest=False).indices
        nn_opp_top2  = dist_opp.topk(2,  dim=2, largest=False).indices
        nn_same1, nn_same2 = nn_same_top2[:, :, 0], nn_same_top2[:, :, 1]
        nn_opp1,  nn_opp2  = nn_opp_top2[:,  :, 0], nn_opp_top2[:,  :, 1]
        b_idx = torch.arange(B, device=pos.device).unsqueeze(1).expand(B, N)
        ball2 = ball_mask.unsqueeze(-1).expand(B, N, 2)

        def gather_rel(nn_idx):
            rp = (pos[b_idx, nn_idx] - pos).masked_fill(ball2, 0.0)
            rv = (vel[b_idx, nn_idx] - vel).masked_fill(ball2, 0.0)
            return rp, rv

        rp_s1, rv_s1 = gather_rel(nn_same1)
        rp_s2, rv_s2 = gather_rel(nn_same2)
        rp_o1, rv_o1 = gather_rel(nn_opp1)
        rp_o2, rv_o2 = gather_rel(nn_opp2)
        T = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        w_same = torch.softmax(-dist_same / T, dim=2)
        w_opp  = torch.softmax(-dist_opp  / T, dim=2)
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)
        att_same_pos = (w_same.unsqueeze(-1) * rel_pos).sum(dim=2).masked_fill(ball2, 0.0)
        att_same_vel = (w_same.unsqueeze(-1) * rel_vel).sum(dim=2).masked_fill(ball2, 0.0)
        att_opp_pos  = (w_opp.unsqueeze(-1)  * rel_pos).sum(dim=2).masked_fill(ball2, 0.0)
        att_opp_vel  = (w_opp.unsqueeze(-1)  * rel_vel).sum(dim=2).masked_fill(ball2, 0.0)
        return torch.cat([rp_s1, rv_s1, rp_s2, rv_s2,
                          rp_o1, rv_o1, rp_o2, rv_o2,
                          att_same_pos, att_same_vel,
                          att_opp_pos,  att_opp_vel], dim=-1)

    def forward(self, X):
        B, _, N, _ = X.shape
        T = self.context_size + self.horizon_size
        ball_mask = (X[:, 0, :, 2] == 0)
        node_type = self._get_node_type(X)
        team_emb  = self.team_emb(node_type)
        h = [torch.zeros(B * N, self.state_dim, device=X.device) for _ in range(self.num_layers)]
        all_preds = []
        prev_pos = None
        pred_mu = None

        for t in range(T):
            if t < self.context_size:
                pos = X[:, t, :, :2]
                vel = torch.zeros_like(pos) if prev_pos is None else pos - prev_pos
                prev_pos = pos
            else:
                pos = pred_mu.reshape(B, N, 2)
                vel = pos - prev_pos
                prev_pos = pos

            ball_pos, ball_vel = self._ball_broadcast(pos, vel, ball_mask)
            motion = torch.cat([pos, vel, pos - ball_pos, vel - ball_vel], dim=-1)
            social = self._social_features(pos, vel, node_type, ball_mask)
            x_in = torch.cat([motion, social, team_emb], dim=-1).reshape(B * N, -1)

            layer_in = x_in
            for i in range(self.num_layers):
                h[i] = self.gru_layers[i](layer_in, h[i])
                h[i] = self.attn_layers[i](
                    h[i].reshape(B, N, self.state_dim), pos, node_type
                ).reshape(B * N, self.state_dim)
                layer_in = h[i]

            if self.context_size - 1 <= t < T - 1:
                out = self.proj(h[-1])
                out_k = out.reshape(B * N, K_MIX, 5)
                log_w = out_k[:, :, 4] - out_k[:, :, 4].logsumexp(dim=-1, keepdim=True)
                w = log_w.exp()
                pred_mu = (w.unsqueeze(-1) * out_k[:, :, :2]).sum(dim=1)
                all_preds.append(out)

        preds = torch.stack(all_preds, dim=0)
        if self.training:
            return preds
        preds_k = preds.reshape(preds.shape[0], B * N, K_MIX, 5)
        log_w = preds_k[:, :, :, 4] - preds_k[:, :, :, 4].logsumexp(dim=-1, keepdim=True)
        w = log_w.exp()
        return (w.unsqueeze(-1) * preds_k[:, :, :, :2]).sum(dim=2)


class Trainer:
    def __init__(self, train_path, val_path, device='cuda', epochs=35, batch_size=32, **kwargs):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

        if TRAIN_TXT_PSEUDO.exists() and VAL_TXT.exists():
            train_files = TRAIN_TXT_PSEUDO.read_text().splitlines()
            val_files   = VAL_TXT.read_text().splitlines()
            print(f'v58_s1: using {len(train_files)} files (real + pseudo) from {TRAIN_TXT_PSEUDO.name}')
        elif TRAIN_TXT.exists() and VAL_TXT.exists():
            train_files = TRAIN_TXT.read_text().splitlines()
            val_files   = VAL_TXT.read_text().splitlines()
        else:
            import os
            train_files = [str(p) for p in Path(train_path).glob('*.pt')]
            val_files   = [str(p) for p in Path(val_path).glob('*.pt')]

        mu, sigma = self._compute_normalization(train_files)
        self.mu = mu
        self.sigma = sigma
        self.train_dataset = NBADataset(train_files, 8, 12, mu, sigma, augment=True)
        self.val_dataset   = NBADataset(val_files,   8, 12, mu, sigma, augment=False)

    def _compute_normalization(self, file_list):
        all_pos = []
        for f in file_list:
            seq = torch.load(f, weights_only=False)
            all_pos.append(seq[:, :, [0, 1]])
        all_pos = torch.cat(all_pos, dim=0)
        return all_pos.mean(dim=(0, 1)), all_pos.std(dim=(0, 1))

    def train(self):
        torch.manual_seed(1)
        random.seed(1)
        generator = torch.Generator('cpu').manual_seed(1)

        train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start, seed=1, shuffle=True)
        val_sampler   = NBASampler(self.batch_size, self.val_dataset.max_start,   seed=1, shuffle=False)
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size,
                              sampler=train_sampler, generator=generator)
        val_dl   = DataLoader(self.val_dataset,   batch_size=self.batch_size,
                              sampler=val_sampler,   generator=generator)

        model = NBAv58Model().to(self.device)
        train_loss_fn = MoGNLLLoss()
        val_loss_fn   = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            model.train()
            self._run_epoch(train_dl, train_loss_fn, model, optimizer, 'train')
            model.train(False)
            with torch.no_grad():
                self._run_epoch(val_dl, val_loss_fn, model, None, 'val')
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            scheduler.step()

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v58_s1.pth')
        self.model = model

    def _run_epoch(self, dataloader, loss_fn, model, optimizer, split):
        avg_loss = 0.0
        for X, y in tqdm(dataloader, desc=f'  {split}'):
            X, y = X.to(self.device), y.to(self.device)
            pred = model(X)
            loss = loss_fn.compute(pred, y)
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            avg_loss += loss.detach()
        avg_loss = avg_loss / len(dataloader)
        print(f'    {split} loss: {avg_loss.item():.4f}')
