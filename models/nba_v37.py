"""v37: Non-autoregressive Transformer.
Encode 8 context frames x 11 players = 88 joint tokens with TransformerEncoder, then decode
all 12 future steps simultaneously per player via MLP. No autoregressive rollout means no
compounding prediction error — the key failure mode of the GRU decoder.
Uses proper train/val split (train_files.txt / val_files.txt). 35 epochs to test paradigm.
"""
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

_DIR = Path(__file__).parent
WEIGHTS_DIR = _DIR / 'weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TXT = _DIR.parent / 'train_files.txt'
VAL_TXT   = _DIR.parent / 'val_files.txt'

TEAM_EMB_DIM = 8
D_MODEL = 128


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


class MultiStepMSE:
    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def compute(self, pred, target):
        B, T, N, F = target.shape
        target = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        return sum(self.loss_fn(pred[t], target[t]) for t in range(T)) / T


class NBAv37Model(nn.Module):
    """Non-autoregressive Transformer: 88 context tokens → encode → MLP decode all 12 steps."""

    def __init__(self, d_model=D_MODEL, context_size=8, horizon_size=12, num_heads=4, num_layers=3):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size

        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        self.input_proj = nn.Linear(8 + TEAM_EMB_DIM, d_model)
        self.time_emb = nn.Embedding(context_size, d_model)
        self.type_emb = nn.Embedding(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Linear(256, horizon_size * 2),
        )

    def forward(self, X):
        B, T, N, _ = X.shape

        pos = X[:, :, :, :2]
        vel = torch.zeros_like(pos)
        vel[:, 1:] = pos[:, 1:] - pos[:, :-1]

        ball_mask_f = (X[:, :, :, 2] == 0).float().unsqueeze(-1)
        ball_pos = (pos * ball_mask_f).sum(2, keepdim=True).expand_as(pos)
        ball_vel = (vel * ball_mask_f).sum(2, keepdim=True).expand_as(vel)

        team = X[:, 0, :, 3]
        node_type = torch.zeros(B, N, dtype=torch.long, device=X.device)
        node_type[team == -1] = 1
        node_type[team == 1] = 2

        team_emb = self.team_emb(node_type).unsqueeze(1).expand(B, T, N, TEAM_EMB_DIM)
        motion = torch.cat([pos, vel, pos - ball_pos, vel - ball_vel], dim=-1)
        features = torch.cat([motion, team_emb], dim=-1)  # [B, T, N, 16]

        tokens = self.input_proj(features.reshape(B, T * N, 8 + TEAM_EMB_DIM))

        t_idx = torch.arange(T, device=X.device).unsqueeze(1).expand(T, N).reshape(T * N)
        tokens = tokens + self.time_emb(t_idx)

        type_exp = node_type.unsqueeze(1).expand(B, T, N).reshape(B, T * N)
        tokens = tokens + self.type_emb(type_exp)

        encoded = self.encoder(tokens)

        last = encoded[:, (T - 1) * N: T * N, :]  # last context frame, [B, N, d_model]
        pred = self.decoder(last)                   # [B, N, 12*2]
        pred = pred.reshape(B, N, self.horizon_size, 2)
        pred = pred.permute(2, 0, 1, 3).reshape(self.horizon_size, B * N, 2)
        return pred  # [12, B*N, 2]


class Trainer:
    def __init__(self, train_path, val_path, device='cuda', epochs=35, batch_size=32, **kwargs):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

        if TRAIN_TXT.exists() and VAL_TXT.exists():
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
        torch.manual_seed(0)
        random.seed(0)
        generator = torch.Generator('cpu').manual_seed(0)

        train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start, shuffle=True)
        val_sampler   = NBASampler(self.batch_size, self.val_dataset.max_start,   shuffle=False)
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size,
                              sampler=train_sampler, generator=generator)
        val_dl   = DataLoader(self.val_dataset,   batch_size=self.batch_size,
                              sampler=val_sampler,   generator=generator)

        model = NBAv37Model().to(self.device)
        loss_fn = MultiStepMSE()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v37.pth')
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            avg_loss += loss.detach()
        avg_loss = avg_loss / len(dataloader)
        print(f'    {split} loss: {avg_loss.item():.4f}')
