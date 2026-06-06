"""v73 — two-stage REFINEMENT model (literature: DynGroupNet / SAnchor refinement improves
deterministic ADE). Stage 1 = FROZEN pretrained v46 (model_v46_300ep.pth) produces the coarse
12-step MoG-mean trajectory. Stage 2 = a Refiner that sees the WHOLE coarse trajectory (all 12
steps, all 11 agents) at once -- non-autoregressive + social -- and outputs per-step corrections.

Why this targets our error: autoregressive rollout compounds (per-step ball error 0.07 -> 8.6 from
t=1..12). The refiner has GLOBAL temporal context (not causal), so it can pull late-step drift back
and enforce trajectory-wide social consistency -- exactly what a causal rollout cannot.

Only the refiner trains (v46 frozen). Loss = ball-weighted, timestep-weighted MSE on the REFINED
trajectory (directly optimizes the competition metric, with our validated ball emphasis).
"""
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import nba_v46
from nba_v46 import (NBAv46Model, TypeAwareDistanceBiasedAttention, NBASampler,
                     MultiStepMSE, WEIGHTS_DIR, TEAM_EMB_DIM)

HOR = 12
COARSE_WEIGHTS = 'model_v46_300ep.pth'


class BallWeightedMSE:
    """Timestep-weighted (1->2) MSE on the refined trajectory, ball (entity 0) upweighted."""
    def __init__(self, ball_w=3.0):
        self.ball_w = ball_w

    def compute(self, pred, target):
        B, T, N, _ = target.shape
        tgt = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        se = ((pred - tgt) ** 2).sum(-1)               # [T, B*N]
        w = torch.ones(N, device=pred.device); w[0] = self.ball_w
        w = w.repeat(B)                                 # [B*N], ball at b*N+0
        tw = torch.linspace(1.0, 2.0, T, device=pred.device)
        per_t = (se * w).sum(1) / w.sum()               # [T]
        return (tw * per_t).sum() / tw.sum()


class Refiner(nn.Module):
    def __init__(self, state_dim=128, num_heads=4):
        super().__init__()
        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        in_dim = HOR * 2 + HOR * 2 + 4 + TEAM_EMB_DIM   # coarse pos + coarse vel + last(pos,vel) + team
        self.enc = nn.Sequential(nn.Linear(in_dim, state_dim), nn.ReLU(),
                                 nn.Linear(state_dim, state_dim), nn.ReLU())
        self.attn = nn.ModuleList([TypeAwareDistanceBiasedAttention(state_dim, num_heads) for _ in range(2)])
        self.dec = nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU(),
                                 nn.Linear(state_dim, HOR * 2))

    def forward(self, coarse, X, node_type):
        # coarse [B, HOR, N, 2] (normalized); X [B, 8, N, F]
        B, H, N, _ = coarse.shape
        cvel = torch.zeros_like(coarse); cvel[:, 1:] = coarse[:, 1:] - coarse[:, :-1]
        last_pos = X[:, -1, :, :2]
        last_vel = X[:, -1, :, :2] - X[:, -2, :, :2]
        team = self.team_emb(node_type)
        feat = torch.cat([
            coarse.permute(0, 2, 1, 3).reshape(B, N, H * 2),
            cvel.permute(0, 2, 1, 3).reshape(B, N, H * 2),
            last_pos, last_vel, team,
        ], dim=-1)
        h = self.enc(feat)                              # [B,N,D]
        for a in self.attn:
            h = a(h, last_pos, node_type)               # social refinement at present geometry
        corr = self.dec(h).reshape(B, N, H, 2).permute(0, 2, 1, 3)  # [B,HOR,N,2]
        return coarse + corr


class NBAv73Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse = NBAv46Model()
        self.coarse.load_state_dict(torch.load(WEIGHTS_DIR / COARSE_WEIGHTS, weights_only=True))
        for p in self.coarse.parameters():
            p.requires_grad = False
        self.refiner = Refiner()

    def _node_type(self, X):
        team = X[:, 0, :, 3]
        nt = torch.zeros_like(team, dtype=torch.long)
        nt[team == -1] = 1
        nt[team == 1] = 2
        return nt

    def forward(self, X):
        B, _, N, _ = X.shape
        self.coarse.train(False)                        # keep frozen coarse in mean-output mode
        with torch.no_grad():
            c = self.coarse(X)                          # [HOR, B*N, 2] MoG mean
        coarse = c.reshape(HOR, B, N, 2).permute(1, 0, 2, 3)        # [B,HOR,N,2]
        refined = self.refiner(coarse, X, self._node_type(X))       # [B,HOR,N,2]
        return refined.permute(1, 0, 2, 3).reshape(HOR, B * N, 2)   # [T, B*N, 2]


class Trainer(nba_v46.Trainer):
    def train(self):
        torch.manual_seed(0); random.seed(0)
        generator = torch.Generator('cpu').manual_seed(0)
        train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start, shuffle=True)
        val_sampler = NBASampler(self.batch_size, self.val_dataset.max_start, shuffle=False)
        train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, generator=generator)
        val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=val_sampler, generator=generator)

        model = NBAv73Model().to(self.device)
        train_loss_fn = BallWeightedMSE(ball_w=3.0)
        val_loss_fn = MultiStepMSE()
        optimizer = torch.optim.Adam(model.refiner.parameters(), lr=1e-3, weight_decay=5e-4)
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

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v73.pth')
        self.model = model
