"""v66 — stronger agent-centric model. Fixes v65's two weak points:
  1. TWO stacked interaction-attention layers (v65 had one) — deeper agent-agent reasoning.
  2. PER-TIMESTEP decoder: each of the 12 future steps gets its own learned time embedding fed
     through a shared MLP (v65 decoded all 12 from a single linear, which is too weak).
Everything else (per-agent local frames, GRU temporal encoder, local-frame MoG NLL loss) reuses
nba_v65. This is the real swing for the top tier — trained to 300ep, judged at convergence
(NOT at 50ep; see the late-bloomer pattern in findings.md).
"""
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import nba_v46
import nba_v65
from nba_v46 import GRUCell, NBASampler, MultiStepMSE, WEIGHTS_DIR
from nba_v65 import (make_frames, pt_to_local, pt_to_global, AgentInteraction,
                     MoGLocalNLLLoss, K_MIX, TEAM_EMB_DIM, CTX, HOR)

TIME_DIM = 16


class NBAv66Model(nn.Module):
    def __init__(self, state_dim=128, num_heads=4, enc_layers=2, interact_layers=2):
        super().__init__()
        self.D = state_dim
        self.enc_layers = enc_layers
        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        self.enc_gru = nn.ModuleList([
            GRUCell(4 if i == 0 else state_dim, state_dim) for i in range(enc_layers)
        ])
        self.interact = nn.ModuleList([
            AgentInteraction(state_dim, num_heads) for _ in range(interact_layers)
        ])
        self.time_emb = nn.Embedding(HOR, TIME_DIM)
        self.dec = nn.Sequential(
            nn.Linear(state_dim + TEAM_EMB_DIM + TIME_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, K_MIX * 5),
        )
        self._frames = None

    def _node_type(self, X):
        team = X[:, 0, :, 3]
        nt = torch.zeros_like(team, dtype=torch.long)
        nt[team == -1] = 1
        nt[team == 1] = 2
        return nt

    def forward(self, X):
        B, _, N, _ = X.shape
        pos_ctx = X[:, :, :, :2]
        c, s, o = make_frames(pos_ctx)
        self._frames = (c, s, o)
        nt = self._node_type(X)
        team_emb = self.team_emb(nt)                                  # [B,N,8]

        c_t = c[:, None, :]; s_t = s[:, None, :]; o_t = o[:, None, :, :]
        traj_local = pt_to_local(pos_ctx, c_t, s_t, o_t)
        vel_local = torch.zeros_like(traj_local)
        vel_local[:, 1:] = traj_local[:, 1:] - traj_local[:, :-1]
        inp = torch.cat([traj_local, vel_local], dim=-1)             # [B,CTX,N,4]

        h = [torch.zeros(B * N, self.D, device=X.device) for _ in range(self.enc_layers)]
        for t in range(CTX):
            layer_in = inp[:, t].reshape(B * N, 4)
            for i in range(self.enc_layers):
                h[i] = self.enc_gru[i](layer_in, h[i])
                layer_in = h[i]
        z = h[-1].reshape(B, N, self.D)

        # geometry in agent-i local frames (computed once, reused by both interaction layers)
        rel = o.unsqueeze(1) - o.unsqueeze(2)
        ci = c[:, :, None]; si = s[:, :, None]
        relpos = torch.stack([ci * rel[..., 0] + si * rel[..., 1],
                              -si * rel[..., 0] + ci * rel[..., 1]], dim=-1)
        vel_pres = pos_ctx[:, -1] - pos_ctx[:, -2]
        rvel = vel_pres.unsqueeze(1) - vel_pres.unsqueeze(2)
        relvel = torch.stack([ci * rvel[..., 0] + si * rvel[..., 1],
                              -si * rvel[..., 0] + ci * rvel[..., 1]], dim=-1)
        dist = relpos.norm(dim=-1, keepdim=True)
        type_pair = nt.unsqueeze(2) * 3 + nt.unsqueeze(1)
        for layer in self.interact:
            z = layer(z, relpos, relvel, dist, type_pair)            # [B,N,D]

        # per-timestep decode: each future step gets its own time embedding
        steps = torch.arange(HOR, device=X.device)
        te = self.time_emb(steps)                                    # [HOR, TIME_DIM]
        zt = z.unsqueeze(2).expand(B, N, HOR, self.D)
        tet = te.view(1, 1, HOR, TIME_DIM).expand(B, N, HOR, TIME_DIM)
        emb = team_emb.unsqueeze(2).expand(B, N, HOR, TEAM_EMB_DIM)
        dec_in = torch.cat([zt, emb, tet], dim=-1)
        out = self.dec(dec_in).reshape(B, N, HOR, K_MIX, 5)

        if self.training:
            return out.permute(2, 0, 1, 3, 4).reshape(HOR, B * N, K_MIX * 5)

        logit = out[..., 4]
        logw = logit - logit.logsumexp(dim=-1, keepdim=True)
        w = logw.exp().unsqueeze(-1)
        mu_local = (w * out[..., :2]).sum(dim=3)
        glob = pt_to_global(mu_local, c[:, :, None], s[:, :, None], o[:, :, None, :])
        return glob.permute(2, 0, 1, 3).reshape(HOR, B * N, 2)


class Trainer(nba_v46.Trainer):
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

        model = NBAv66Model().to(self.device)
        train_loss_fn = MoGLocalNLLLoss()
        val_loss_fn = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            model.train()
            avg = 0.0
            for X, y in train_dl:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                loss = train_loss_fn.compute(pred, y, model._frames)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                avg += loss.detach()
            print(f'    train loss: {(avg / len(train_dl)).item():.4f}')

            model.train(False)
            with torch.no_grad():
                vavg = 0.0
                for X, y in val_dl:
                    X, y = X.to(self.device), y.to(self.device)
                    vavg += val_loss_fn.compute(model(X), y).detach()
                print(f'    val loss: {(vavg / len(val_dl)).item():.4f}')

            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            scheduler.step()

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v66.pth')
        self.model = model
