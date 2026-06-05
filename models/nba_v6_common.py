"""Shared scaffolding for v61/v62/v63 — cheap structural variants of v46 (see BALL_ERROR_PLAN.md
and the architecture brainstorm). Each variant changes ONE thing vs v46 so the A/B is clean:

  v61  anchored_forward(anchor='pos')     -> head predicts DISPLACEMENT (pos + delta)
  v62  anchored_forward(anchor='posvel')  -> head predicts residual over CONSTANT-VELOCITY (pos+vel+delta)
  v63  rotate=True                        -> train-time random rotation aug (toward rot-equivariance)

Everything else (arch, MoG K=2 loss, 300ep/T_max, lr, wd, val MSE, x/y-flip aug) is identical to v46.
"""
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import nba_v46
from nba_v46 import (NBASampler, MoGNLLLoss, MultiStepMSE, NBAv46Model,
                     NBADataset, K_MIX, WEIGHTS_DIR)


def anchored_forward(self, X, anchor='none'):
    """v46.forward, but the head's mu can be interpreted relative to an anchor.
    anchor='none'   -> absolute position (identical to v46)
    anchor='pos'    -> displacement from current position
    anchor='posvel' -> residual over constant-velocity extrapolation (pos + vel)
    """
    B, _, N, _ = X.shape
    T = self.context_size + self.horizon_size
    ball_mask = (X[:, 0, :, 2] == 0)
    node_type = self._get_node_type(X)
    team_emb = self.team_emb(node_type)
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
            if anchor != 'none':
                a = pos.reshape(B * N, 1, 2)
                if anchor == 'posvel':
                    a = a + vel.reshape(B * N, 1, 2)
                out_k = out_k.clone()
                out_k[:, :, :2] = out_k[:, :, :2] + a
                out = out_k.reshape(B * N, K_MIX * 5)
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


class NBAv61Model(NBAv46Model):
    """Displacement head: predict pos + delta."""
    def forward(self, X):
        return anchored_forward(self, X, 'pos')


class NBAv62Model(NBAv46Model):
    """Constant-velocity residual head: predict pos + vel + delta."""
    def forward(self, X):
        return anchored_forward(self, X, 'posvel')


class RotNBADataset(NBADataset):
    """v46 dataset + random rotation (after the x/y flips). Normalized coords rotate about origin."""
    def __getitem__(self, index):
        seq_idx, start = index
        X = self.sequences[seq_idx][start:start + self.context_size].clone()
        y = self.sequences[seq_idx][start + self.context_size:start + self.window_size].clone()
        if self.augment:
            if random.random() < 0.5:
                X[:, :, 0] = -X[:, :, 0]; y[:, :, 0] = -y[:, :, 0]
            if random.random() < 0.5:
                X[:, :, 1] = -X[:, :, 1]; y[:, :, 1] = -y[:, :, 1]
            th = (random.random() * 2 - 1) * (math.pi / 6)  # +/- 30 deg
            c, s = math.cos(th), math.sin(th)
            for t in (X, y):
                x0 = t[:, :, 0].clone()
                y0 = t[:, :, 1].clone()
                t[:, :, 0] = c * x0 - s * y0
                t[:, :, 1] = s * x0 + c * y0
        return X, y


def make_trainer(model_factory, save_stem, rotate=False):
    class _V6Trainer(nba_v46.Trainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if rotate:
                # reuse already-loaded sequences; just swap the __getitem__ behavior
                self.train_dataset.__class__ = RotNBADataset

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

            model = model_factory().to(self.device)
            train_loss_fn = MoGNLLLoss()
            val_loss_fn = MultiStepMSE()
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

            torch.save(model.state_dict(), WEIGHTS_DIR / f'{save_stem}.pth')
            self.model = model

    return _V6Trainer
