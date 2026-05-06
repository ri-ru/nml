"""v14: Directional attention — use full 2D relative position (direction + distance) as attention bias.
V12 is the new best (4.1568): nearest-social + ball-relative features + typed attention.
Current attention bias uses only scalar distance → symmetric, cannot distinguish left/right.
Fix: project (dx, dy, dist) → num_heads instead of just (dist,) → num_heads.
This lets attention differentiate "player 5ft to my right" vs "player 5ft to my left",
which matters for predicting who will receive a pass or who is driving baseline.
Architecture: v12 features + directional dist_proj (Linear(3, num_heads) instead of Linear(1, num_heads)).
Same state_dim=128, 2 layers, lr=1e-3, 35 epochs.
"""
import os
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

TEAM_EMB_DIM = 8


class NBADataset(Dataset):
    def __init__(self, data_path, context_size, horizon_size, mu, sigma):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.sequences = []
        self.max_start = []
        for f in [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')]:
            seq = torch.load(f, weights_only=False)
            seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - mu) / sigma
            self.sequences.append(seq)
            self.max_start.append(max(0, len(seq) - self.window_size))

    def __getitem__(self, index):
        seq_idx, start = index
        X = self.sequences[seq_idx][start:start + self.context_size]
        y = self.sequences[seq_idx][start + self.context_size:start + self.window_size]
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


class DirectionalAttention(nn.Module):
    """Typed attention with directional bias: projects (dx, dy, dist) → heads instead of just dist."""

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
        # 3 inputs: dx, dy (normalized direction) + dist (scalar) → num_heads
        self.dir_proj = nn.Linear(3, num_heads)
        self.type_pair_bias = nn.Embedding(num_types * num_types, num_heads)

    def forward(self, h, pos, node_type):
        B, N, D = h.shape
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)          # [B, N, N, 2]
        dist = diff.norm(dim=-1, keepdim=True)              # [B, N, N, 1]
        direction = diff / (dist + 1e-6)                    # [B, N, N, 2] unit vector
        dir_feat = torch.cat([direction, dist], dim=-1)     # [B, N, N, 3]
        dir_bias = self.dir_proj(dir_feat).permute(0, 3, 1, 2)  # [B, heads, N, N]

        type_i = node_type.unsqueeze(2).expand(-1, -1, N)
        type_j = node_type.unsqueeze(1).expand(-1, N, -1)
        type_bias = self.type_pair_bias(type_i * 3 + type_j).permute(0, 3, 1, 2)

        def split_heads(t):
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(self.q(h)), split_heads(self.k(h)), split_heads(self.v(h))
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale + dir_bias + type_bias, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.norm(h + self.out(out))


class NBAv14Model(nn.Module):
    """V12 features + directional attention (dx,dy,dist) replacing scalar-distance bias."""

    def __init__(self, state_dim, context_size, horizon_size, num_heads=4, num_layers=2):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.state_dim = state_dim
        self.num_layers = num_layers

        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        # v12 feature set: 16 (motion + ball-rel + nearest-social) + TEAM_EMB_DIM
        input_dim_0 = 16 + TEAM_EMB_DIM

        self.gru_layers = nn.ModuleList([
            GRUCell(input_dim_0 if i == 0 else state_dim, state_dim)
            for i in range(num_layers)
        ])
        self.attn_layers = nn.ModuleList([
            DirectionalAttention(state_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.proj = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),
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

    def _nearest_social_features(self, pos, vel, node_type, ball_mask):
        B, N, _ = pos.shape
        INF = 1e9
        dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)
        eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
        dist = dist.masked_fill(eye, INF)

        type_i = node_type.unsqueeze(2)
        type_j = node_type.unsqueeze(1)

        same_mask = (type_i == type_j) & (type_i != 0) & ~eye
        nn_same = dist.masked_fill(~same_mask, INF).argmin(dim=2)

        opp_mask = ((type_i == 1) & (type_j == 2)) | ((type_i == 2) & (type_j == 1))
        nn_opp = dist.masked_fill(~opp_mask, INF).argmin(dim=2)

        b = torch.arange(B, device=pos.device).unsqueeze(1).expand(B, N)
        ball2 = ball_mask.unsqueeze(-1).expand(B, N, 2)
        rel_same_pos = (pos[b, nn_same] - pos).masked_fill(ball2, 0.0)
        rel_same_vel = (vel[b, nn_same] - vel).masked_fill(ball2, 0.0)
        rel_opp_pos = (pos[b, nn_opp] - pos).masked_fill(ball2, 0.0)
        rel_opp_vel = (vel[b, nn_opp] - vel).masked_fill(ball2, 0.0)

        return torch.cat([rel_same_pos, rel_same_vel, rel_opp_pos, rel_opp_vel], dim=-1)

    def forward(self, X):
        B, _, N, _ = X.shape
        T = self.context_size + self.horizon_size

        ball_mask = (X[:, 0, :, 2] == 0)
        node_type = self._get_node_type(X)
        team_emb = self.team_emb(node_type)

        h = [torch.zeros(B * N, self.state_dim, device=X.device) for _ in range(self.num_layers)]
        all_preds = []
        prev_pos = None
        x = None

        for t in range(T):
            if t < self.context_size:
                pos = X[:, t, :, :2]
                vel = torch.zeros_like(pos) if prev_pos is None else pos - prev_pos
                prev_pos = pos
            else:
                pos = x.reshape(B, N, 2)
                vel = pos - prev_pos
                prev_pos = pos

            ball_pos, ball_vel = self._ball_broadcast(pos, vel, ball_mask)
            motion = torch.cat([pos, vel, pos - ball_pos, vel - ball_vel], dim=-1)
            social = self._nearest_social_features(pos, vel, node_type, ball_mask)
            x_in = torch.cat([motion, social, team_emb], dim=-1).reshape(B * N, -1)

            layer_in = x_in
            for i in range(self.num_layers):
                h[i] = self.gru_layers[i](layer_in, h[i])
                h[i] = self.attn_layers[i](
                    h[i].reshape(B, N, self.state_dim), pos, node_type
                ).reshape(B * N, self.state_dim)
                layer_in = h[i]

            if self.context_size - 1 <= t < T - 1:
                x = self.proj(h[-1])
                all_preds.append(x)

        return torch.stack(all_preds, dim=0)


class Trainer:
    def __init__(self, train_path, val_path, device='cuda', epochs=35, batch_size=32, **kwargs):
        self.train_path = train_path
        self.val_path = val_path
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

        mu, sigma = self._compute_normalization(train_path)
        self.mu = mu
        self.sigma = sigma
        self.train_dataset = NBADataset(train_path, 8, 12, mu, sigma)
        self.val_dataset = NBADataset(val_path, 8, 12, mu, sigma)

    def _compute_normalization(self, data_path):
        all_pos = []
        for f in [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')]:
            seq = torch.load(f, weights_only=False)
            all_pos.append(seq[:, :, [0, 1]])
        all_pos = torch.cat(all_pos, dim=0)
        return all_pos.mean(dim=(0, 1)), all_pos.std(dim=(0, 1))

    def train(self):
        torch.manual_seed(0)
        random.seed(0)
        generator = torch.Generator('cpu').manual_seed(0)

        train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start, shuffle=True)
        val_sampler = NBASampler(self.batch_size, self.val_dataset.max_start, shuffle=False)
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size,
                              sampler=train_sampler, generator=generator)
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size,
                            sampler=val_sampler, generator=generator)

        model = NBAv14Model(128, 8, 12, num_heads=4, num_layers=2).to(self.device)
        loss_fn = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            model.train()
            self._run_epoch(train_dl, loss_fn, model, optimizer, 'train')
            model.train(False)
            with torch.no_grad():
                self._run_epoch(val_dl, loss_fn, model, None, 'val')
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            scheduler.step()

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v14.pth')
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
