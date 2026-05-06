"""v21: GNN edge-MLP message passing replaces typed distance-biased attention.
V20 (4.0292 best): typed attention uses distance+type as scalar biases on Q@K^T.
Hypothesis: proper graph network layers — edge_MLP(h_i, h_j, rel_pos, rel_vel, type_emb)
→ messages → distance-weighted aggregation → node update — are more expressive because
the edge function directly ingests continuous geometry (not just scalar dist bias),
letting the model learn direction-aware interaction patterns without NaN risk
(raw rel_pos fed to MLP, no normalization). Same v20 input features, same state_dim=128.
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


class GNNMessagePass(nn.Module):
    """Edge-MLP message passing: m_ij = MLP(h_i, h_j, rel_pos, rel_vel, type_emb)."""

    def __init__(self, state_dim, num_types=3):
        super().__init__()
        self.state_dim = state_dim
        # type pair embedding: 3 types → 3x3=9 pairs
        self.type_pair_emb = nn.Embedding(num_types * num_types, 4)
        # edge: h_i(D) + h_j(D) + rel_pos(2) + rel_vel(2) + type_pair(4) = 2D+8
        edge_in = 2 * state_dim + 8
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )
        # node update: h(D) + aggregated_msg(D) → D
        self.node_update = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, h, pos, vel, node_type):
        B, N, D = h.shape

        # Relative geometry: raw (no normalization → no NaN)
        rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1)   # [B, N, N, 2]: pos_j - pos_i
        rel_vel = vel.unsqueeze(2) - vel.unsqueeze(1)   # [B, N, N, 2]
        dist = rel_pos.norm(dim=-1)                      # [B, N, N]

        # Type pair embedding
        type_i = node_type.unsqueeze(2).expand(-1, -1, N)
        type_j = node_type.unsqueeze(1).expand(-1, N, -1)
        tp_emb = self.type_pair_emb(type_i * 3 + type_j)  # [B, N, N, 4]

        # Edge messages: m[b, i, j] = message from j to i
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        edge_in = torch.cat([h_i, h_j, rel_pos, rel_vel, tp_emb], dim=-1)
        messages = self.edge_mlp(edge_in)  # [B, N, N, D]

        # Distance-weighted aggregation (soft; self excluded via dist=0 → large negative)
        eye = torch.eye(N, dtype=torch.bool, device=h.device).unsqueeze(0)
        logits = -dist.masked_fill(eye, 1e4)
        weights = torch.softmax(logits, dim=2)           # [B, N, N]
        agg = (weights.unsqueeze(-1) * messages).sum(dim=2)  # [B, N, D]

        # Node update with residual
        h_new = self.node_update(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class NBAv21Model(nn.Module):
    """GNN edge-MLP message passing layers (replace typed distance-biased attention)."""

    def __init__(self, state_dim, context_size, horizon_size, num_layers=2):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.state_dim = state_dim
        self.num_layers = num_layers

        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        # v20 features: motion(8) + K1-hard(8) + soft-global(8) + team_emb(8) = 32
        input_dim_0 = 24 + TEAM_EMB_DIM

        self.log_temperature = nn.Parameter(torch.zeros(1))

        self.gru_layers = nn.ModuleList([
            GRUCell(input_dim_0 if i == 0 else state_dim, state_dim)
            for i in range(num_layers)
        ])
        self.gnn_layers = nn.ModuleList([
            GNNMessagePass(state_dim)
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

    def _social_features(self, pos, vel, node_type, ball_mask):
        """K=1 hard-nearest + soft global attention (same as v20)."""
        B, N, _ = pos.shape
        INF = 1e4

        dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)
        eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)

        type_i = node_type.unsqueeze(2)
        type_j = node_type.unsqueeze(1)
        same_mask = (type_i == type_j) & (type_i != 0) & ~eye
        opp_mask = ((type_i == 1) & (type_j == 2)) | ((type_i == 2) & (type_j == 1))

        dist_same = dist.masked_fill(~same_mask, INF)
        dist_opp = dist.masked_fill(~opp_mask, INF)

        nn_same1 = dist_same.argmin(dim=2)
        nn_opp1 = dist_opp.argmin(dim=2)

        b_idx = torch.arange(B, device=pos.device).unsqueeze(1).expand(B, N)
        ball2 = ball_mask.unsqueeze(-1).expand(B, N, 2)

        def gather_rel(nn_idx):
            rp = (pos[b_idx, nn_idx] - pos).masked_fill(ball2, 0.0)
            rv = (vel[b_idx, nn_idx] - vel).masked_fill(ball2, 0.0)
            return rp, rv

        rp_s1, rv_s1 = gather_rel(nn_same1)
        rp_o1, rv_o1 = gather_rel(nn_opp1)

        T = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        w_same = torch.softmax(-dist_same / T, dim=2)
        w_opp = torch.softmax(-dist_opp / T, dim=2)

        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)

        att_same_pos = (w_same.unsqueeze(-1) * rel_pos).sum(dim=2).masked_fill(ball2, 0.0)
        att_same_vel = (w_same.unsqueeze(-1) * rel_vel).sum(dim=2).masked_fill(ball2, 0.0)
        att_opp_pos = (w_opp.unsqueeze(-1) * rel_pos).sum(dim=2).masked_fill(ball2, 0.0)
        att_opp_vel = (w_opp.unsqueeze(-1) * rel_vel).sum(dim=2).masked_fill(ball2, 0.0)

        return torch.cat([rp_s1, rv_s1, rp_o1, rv_o1,
                          att_same_pos, att_same_vel,
                          att_opp_pos, att_opp_vel], dim=-1)  # [B, N, 16]

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
            social = self._social_features(pos, vel, node_type, ball_mask)
            x_in = torch.cat([motion, social, team_emb], dim=-1).reshape(B * N, -1)

            layer_in = x_in
            for i in range(self.num_layers):
                h[i] = self.gru_layers[i](layer_in, h[i])
                h[i] = self.gnn_layers[i](
                    h[i].reshape(B, N, self.state_dim), pos, vel, node_type
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

        model = NBAv21Model(128, 8, 12, num_layers=2).to(self.device)
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

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v21.pth')
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
