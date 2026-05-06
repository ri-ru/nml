"""v5: Scheduled teacher forcing + larger GRU + dropout.
Same velocity-augmented GRU + distance-biased attention as v3 (best so far),
but with state_dim=192, 3 layers, dropout=0.1, and teacher forcing (0.8→0.0)
to eliminate error accumulation during training. CosineAnnealing LR.
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


class DistanceBiasedAttention(nn.Module):
    def __init__(self, state_dim, num_heads=4):
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

    def forward(self, h, pos):
        B, N, D = h.shape
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = diff.norm(dim=-1, keepdim=True)
        dist_bias = self.dist_proj(dist).permute(0, 3, 1, 2)

        def split_heads(t):
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(self.q(h)), split_heads(self.k(h)), split_heads(self.v(h))
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale + dist_bias, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.norm(h + self.out(out))


class NBAv5Model(nn.Module):
    """Larger GRU + dist-biased attention with scheduled teacher forcing."""

    def __init__(self, state_dim, context_size, horizon_size, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.state_dim = state_dim
        self.num_layers = num_layers

        self.gru_layers = nn.ModuleList([
            GRUCell(6 if i == 0 else state_dim, state_dim)
            for i in range(num_layers)
        ])
        self.attn_layers = nn.ModuleList([
            DistanceBiasedAttention(state_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, X, y_teacher=None, teacher_forcing_ratio=0.0):
        B, _, N, _ = X.shape
        T = self.context_size + self.horizon_size
        h = [torch.zeros(B * N, self.state_dim, device=X.device) for _ in range(self.num_layers)]
        all_preds = []
        prev_pos = None
        x = None

        for t in range(T):
            if t < self.context_size:
                pos = X[:, t, :, :2]
                static = X[:, t, :, 2:]
                vel = torch.zeros_like(pos) if prev_pos is None else pos - prev_pos
                prev_pos = pos
                x_in = torch.cat([pos, vel, static], dim=-1).reshape(B * N, 6)
            else:
                future_t = t - self.context_size
                if (y_teacher is not None and teacher_forcing_ratio > 0
                        and random.random() < teacher_forcing_ratio):
                    pred_pos = y_teacher[:, future_t, :, :2]
                else:
                    pred_pos = x.reshape(B, N, 2)
                vel = pred_pos - prev_pos
                prev_pos = pred_pos
                pos = pred_pos
                static = X[:, 0, :, 2:]
                x_in = torch.cat([pred_pos, vel, static], dim=-1).reshape(B * N, 6)

            layer_in = x_in
            for i in range(self.num_layers):
                h[i] = self.gru_layers[i](layer_in, h[i])
                h[i] = self.attn_layers[i](
                    h[i].reshape(B, N, self.state_dim), pos
                ).reshape(B * N, self.state_dim)
                layer_in = self.dropout(h[i]) if i < self.num_layers - 1 else h[i]

            if self.context_size - 1 <= t < T - 1:
                x = self.proj(h[-1])
                all_preds.append(x)

        return torch.stack(all_preds, dim=0)  # [horizon_size, B*N, 2]


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

        model = NBAv5Model(192, 8, 12, num_heads=4, num_layers=3).to(self.device)
        loss_fn = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            tf_ratio = 0.8 * max(0.0, 1.0 - epoch / self.epochs)
            print(f'Epoch {epoch + 1}/{self.epochs} (tf={tf_ratio:.2f})')
            model.train()
            self._run_epoch(train_dl, loss_fn, model, optimizer, 'train', tf_ratio)
            model.train(False)
            with torch.no_grad():
                self._run_epoch(val_dl, loss_fn, model, None, 'val', 0.0)
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            scheduler.step()

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v5.pth')
        self.model = model

    def _run_epoch(self, dataloader, loss_fn, model, optimizer, split, teacher_forcing_ratio=0.0):
        avg_loss = 0.0
        for X, y in tqdm(dataloader, desc=f'  {split}'):
            X, y = X.to(self.device), y.to(self.device)
            pred = model(X, y_teacher=(y if split == 'train' else None),
                         teacher_forcing_ratio=teacher_forcing_ratio)
            loss = loss_fn.compute(pred, y)
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            avg_loss += loss.detach()
        avg_loss = avg_loss / len(dataloader)
        print(f'    {split} loss: {avg_loss.item():.4f}')
