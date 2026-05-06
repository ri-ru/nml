from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from pathlib import Path
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
import wandb

_DIR = Path(__file__).parent
WEIGHTS_DIR = _DIR / 'weights'
SOLUTIONS_DIR = _DIR / 'solutions'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data classes (unchanged) ──────────────────────────────────────────────────

class NBADataset(Dataset):
    def __init__(self, data_path, context_size, horizon_size, mu, sigma):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.load_data(data_path, mu, sigma)

    def load_data(self, data_path, mu, sigma):
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
        if pred.size(1) != target.size(1):
            raise ValueError(f'Dimension mismatch: {pred.size(1)} vs {target.size(1)}')
        return sum(self.loss_fn(pred[t], target[t]) for t in range(T)) / T


# ── Model ─────────────────────────────────────────────────────────────────────

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
    """Multi-head attention where scores are biased by pairwise player distances.

    Players closer together attend more strongly to each other.
    """

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
        # h: [B, N, D]   pos: [B, N, 2]
        B, N, D = h.shape
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)          # [B, N, N, 2]
        dist = diff.norm(dim=-1, keepdim=True)               # [B, N, N, 1]
        dist_bias = self.dist_proj(dist).permute(0, 3, 1, 2) # [B, heads, N, N]

        def split_heads(t):
            return t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(self.q(h)), split_heads(self.k(h)), split_heads(self.v(h))
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale + dist_bias, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.norm(h + self.out(out))


class NBAv2Model(nn.Module):
    """Stacked GRU + Distance-biased Attention for NBA trajectory prediction.

    Each timestep:
      For each layer:
        1. GRU updates hidden state.
        2. Distance-biased attention: nearby players influence each other more.
    """

    def __init__(self, input_dim, output_dim, state_dim, context_size, horizon_size,
                 num_heads=4, num_layers=2):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.state_dim = state_dim
        self.num_layers = num_layers

        self.gru_layers = nn.ModuleList([
            GRUCell(input_dim if i == 0 else state_dim, state_dim)
            for i in range(num_layers)
        ])
        self.attn_layers = nn.ModuleList([
            DistanceBiasedAttention(state_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.proj = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, X):
        B, _, N, F = X.shape
        T = self.context_size + self.horizon_size
        h = [torch.zeros(B * N, self.state_dim, device=X.device) for _ in range(self.num_layers)]
        all_preds = []
        x = None

        for t in range(T):
            if t < self.context_size:
                x_in = X[:, t, :, :].reshape(B * N, F)
                pos = X[:, t, :, :2]                          # [B, N, 2]
            else:
                x_in = torch.cat([x, X[:, 0, :, 2:].reshape(B * N, 2)], dim=1)
                pos = x.reshape(B, N, 2)                      # predicted positions

            layer_in = x_in
            for i in range(self.num_layers):
                h[i] = self.gru_layers[i](layer_in, h[i])
                h[i] = self.attn_layers[i](
                    h[i].reshape(B, N, self.state_dim), pos
                ).reshape(B * N, self.state_dim)
                layer_in = h[i]

            if self.context_size - 1 <= t < T - 1:
                x = self.proj(h[-1])
                all_preds.append(x)

        return torch.stack(all_preds, dim=0)


# ── Trainer ───────────────────────────────────────────────────────────────────

class NBAv2Trainer:

    ENTITY_MAPPING = {-1: 'Team_A', 0: 'Ball', 1: 'Team_B'}

    def __init__(self, batch_size, epochs, train_path, val_path,
                 device='cuda', context_size=8, horizon_size=12, seed=0,
                 state_dim=128, num_heads=4, num_layers=2):
        self.batch_size = batch_size
        self.context_size = context_size
        self.device = device
        self.epochs = epochs
        self.horizon_size = horizon_size
        self.train_path = train_path
        self.val_path = val_path
        self.seed = seed
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        mu, sigma = self._compute_normalization(train_path)
        self.mu, self.sigma = mu, sigma
        self.train_dataset = NBADataset(train_path, context_size, horizon_size, mu, sigma)
        self.val_dataset = NBADataset(val_path, context_size, horizon_size, mu, sigma)

    def _compute_normalization(self, data_path):
        all_pos = []
        for f in [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pt')]:
            seq = torch.load(f, weights_only=False)
            all_pos.append(seq[:, :, [0, 1]])
        all_pos = torch.cat(all_pos, dim=0)
        return all_pos.mean(dim=(0, 1)), all_pos.std(dim=(0, 1))

    def _set_seed(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        return torch.Generator('cpu').manual_seed(self.seed)

    def pipeline(self):
        sweep_config = {
            'method': 'grid',
            'metric': {'name': 'val_loss', 'goal': 'minimize'},
            'parameters': {'epochs': {'value': self.epochs}, 'batch_size': {'value': self.batch_size}},
        }
        sweep_id = wandb.sweep(sweep_config, project='NML_v2')
        wandb.agent(sweep_id, self.train)

    def train(self):
        generator = self._set_seed()
        train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start, shuffle=True)
        val_sampler = NBASampler(self.batch_size, self.val_dataset.max_start, shuffle=False)
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, generator=generator)
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=val_sampler, generator=generator)

        model = NBAv2Model(4, 2, self.state_dim, self.context_size, self.horizon_size,
                           self.num_heads, self.num_layers).to(self.device)
        loss_fn = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

        with wandb.init(settings=wandb.Settings(_disable_stats=True, _disable_meta=True)):
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
            torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v2.pth')

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
        wandb.log({f'{split}_loss': avg_loss.item()})
        print(f'    {split} loss: {avg_loss.item():.4f}')

    def get_trajectory(self, X):
        model = NBAv2Model(4, 2, self.state_dim, 8, 12, self.num_heads, self.num_layers).to(self.device)
        model.load_state_dict(torch.load(WEIGHTS_DIR / 'model_v2.pth', weights_only=True))
        model.train(False)
        with torch.no_grad():
            pred = model(X.unsqueeze(0).to(self.device)).cpu()
        X = X.clone()
        X[:, :, :2] = X[:, :, :2] * self.sigma + self.mu
        pred = pred * self.sigma + self.mu
        static = X[-1, :, 2:].unsqueeze(0).repeat(pred.size(0), 1, 1)
        pred = torch.cat([pred, static], dim=-1)
        return torch.cat([X, pred], dim=0).detach()

    def get_kaggle_submission(self, test_dir, target_dir=None):
        all_traj = []
        for f in [f for f in os.listdir(test_dir) if f.endswith('.pt')]:
            seq = torch.load(os.path.join(test_dir, f), weights_only=False)
            seq[:, :, [0, 1]] = (seq[:, :, [0, 1]].clone() - self.mu) / self.sigma
            traj = self.get_trajectory(seq)
            traj = traj[8:, :, :2].reshape(-1)
            all_traj.append([int(f.removesuffix('.pt'))] + traj.tolist())
        cols = ['id'] + [f'entity_{i}_time_{t}_{axis}' for t in range(12) for i in range(11) for axis in ['x', 'y']]
        pd.DataFrame(all_traj, columns=cols).set_index('id').sort_index().to_csv(
            SOLUTIONS_DIR / 'solution_v2.csv'
        )
