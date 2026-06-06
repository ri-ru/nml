"""v60 base — Phase 1 of the ball-error attack (see BALL_ERROR_PLAN.md).

Identical to v46 in every way EXCEPT the training loss: the per-(timestep) MoG NLL is additionally
weighted per entity, with the ball (entity index 0, always) upweighted by BALL_W. Everything else
(arch=NBAv46Model, 300ep, T_max=300, lr=1e-3, wd=5e-4, augment, val=MultiStepMSE) is unchanged so
the only variable is the ball weight.

Thin variant files (nba_v60_w2.py, ...) set BALL_W and a checkpoint stem, then do:
    Trainer = build_trainer_class(BALL_W, 'model_v60_w2')
run_one.py discovers `Trainer` and additionally saves model_v60_wN_<epochs>ep.pth.
"""
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: F401 (kept for parity / potential use)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import nba_v46
from nba_v46 import NBAv46Model, NBASampler, MultiStepMSE

K_MIX = nba_v46.K_MIX
WEIGHTS_DIR = nba_v46.WEIGHTS_DIR


class BallWeightedMoGNLLLoss:
    """v46 MoG NLL, but each entity's NLL is scaled; ball (entity 0) by `ball_w`."""

    def __init__(self, ball_w: float):
        self.ball_w = ball_w

    def compute(self, pred, target):
        B, T, N, _ = target.shape
        tgt = target[:, :, :, :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        pred = pred.reshape(T, B * N, K_MIX, 5)
        mu = pred[:, :, :, :2]
        log_sigma = pred[:, :, :, 2:4].clamp(-3, 3)
        logit_w = pred[:, :, :, 4]
        log_w = logit_w - logit_w.logsumexp(dim=-1, keepdim=True)
        var = (log_sigma * 2).exp()
        tgt_k = tgt.unsqueeze(2).expand_as(mu)
        diff = tgt_k - mu
        log_prob_k = -0.5 * (diff ** 2 / var + log_sigma * 2 + math.log(2 * math.pi))
        log_prob_k = log_prob_k.sum(dim=-1)
        log_mix = torch.logsumexp(log_w + log_prob_k, dim=-1)  # [T, B*N]
        nll = -log_mix                                          # [T, B*N]

        # reshape order is [b0n0, b0n1, ..., b0n10, b1n0, ...]; entity n at pos b*N+n.
        # ball is always entity index 0 -> upweight every (b*N + 0) position.
        w_ent = torch.ones(N, device=pred.device)
        w_ent[0] = self.ball_w
        w_ent = w_ent.repeat(B)  # [B*N]
        denom = w_ent.sum()

        t_weights = torch.linspace(1.0, 2.0, T, device=pred.device)
        total = sum(t_weights[t] * (w_ent * nll[t]).sum() / denom for t in range(T))
        return total / t_weights.sum()


def build_trainer_class(ball_w: float, save_stem: str, seed: int = 0):
    """Return a Trainer subclass of v46's Trainer using the ball-weighted loss.
    seed varies weight init + sampler order for ensemble-diversity seed variants."""

    class _BallWeightedTrainer(nba_v46.Trainer):
        def train(self):
            torch.manual_seed(seed)
            random.seed(seed)
            generator = torch.Generator('cpu').manual_seed(seed)

            train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start, seed=seed, shuffle=True)
            val_sampler = NBASampler(self.batch_size, self.val_dataset.max_start, shuffle=False)
            train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  sampler=train_sampler, generator=generator)
            val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                sampler=val_sampler, generator=generator)

            model = NBAv46Model().to(self.device)
            train_loss_fn = BallWeightedMoGNLLLoss(ball_w)
            val_loss_fn = MultiStepMSE()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            for epoch in range(self.epochs):
                print(f'Epoch {epoch + 1}/{self.epochs}  (ball_w={ball_w})')
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

    return _BallWeightedTrainer
