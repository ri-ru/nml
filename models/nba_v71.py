"""v71 — tail-weighted / hard-example ball loss (focal-style on the ball only).

Motivation (BALL_DEEP_ANALYSIS.md): ball error is a HEAVY TAIL — the top ~10% of
windows carry 50% of all ball error, and the best 50% of ball windows are already
player-level (MSE 2.33). v60's UNIFORM ball-weight (BallWeightedMoGNLLLoss, ball_w=8)
upweights ALL ball positions equally, which is wasteful: 90% of ball cases are already
fine. v71 instead concentrates the ball's loss budget onto the HARD ball positions.

Everything else is identical to v46 (same trunk NBAv46Model, MoG K=2, 300/Nep schedule,
lr=1e-3, wd=5e-4, augment, val=MultiStepMSE, timestep linspace(1->2)) so the ONLY variable
vs v60_w8 is uniform-vs-tail redistribution of the ball weight.

=== Loss formula (TailBallMoGNLLLoss) ===
Per timestep t (in [0,T)), per B*N position p:
  nll[t,p]      = -logsumexp_k( log_w + Gaussian_logprob_k )         # v46 MoG NLL, unchanged
For BALL positions (entity index 0, i.e. p % N == 0) only:
  hardness[t,b] = || mu_mix[t,b] - tgt[t,b] ||^2   (DETACHED)        # sq err of MoG mean
  raw[t,b]      = (hardness[t,b] / mean_b hardness[t,·]) ** gamma    # relative hardness
  raw           = clamp(raw, 0, clamp_max)
  w_ball[t,b]   = base_ball_w * raw[t,b] / mean_b raw[t,·]           # mean ball weight == base_ball_w
Player positions: weight = 1. Then the v46 timestep weighting + normalization is applied:
  total = sum_t t_weight[t] * (w_ent[t] * nll[t]).sum() / w_ent[t].sum()   / sum_t t_weight[t]

Because w_ball is renormalized so its per-timestep MEAN equals base_ball_w, the TOTAL ball
loss budget matches v60_w<base_ball_w> exactly; v71 only REDISTRIBUTES that budget toward the
hard tail. Setting base_ball_w=8 makes this a clean A/B vs v60_w8 (same mean ball weight).

Stability: hardness is detached (no grad through the weight). raw is clamped to clamp_max so a
single catastrophic window can't dominate; mean-normalization keeps the loss scale fixed
regardless of gamma. base_ball_w is the only thing inflating loss magnitude (same as v60).

Thin sweep variants set GAMMA / BASE_BALL_W / stem and call build_trainer_class(...).
This top-level file itself trains gamma=1.0, base_ball_w=8 (the headline config).
"""
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import nba_v46
from nba_v46 import NBAv46Model, NBASampler, MultiStepMSE

K_MIX = nba_v46.K_MIX
WEIGHTS_DIR = nba_v46.WEIGHTS_DIR


class TailBallMoGNLLLoss:
    """v46 MoG NLL with focal-style tail-reweighting of the BALL (entity 0) only.

    base_ball_w : mean ball weight (matches v60's uniform ball_w for a clean A/B).
    gamma       : focal exponent on relative hardness (0 => uniform == v60; >0 => tail).
    clamp_max   : cap on relative hardness weight before mean-normalization (stability).
    """

    def __init__(self, base_ball_w: float = 8.0, gamma: float = 1.0, clamp_max: float = 20.0):
        self.base_ball_w = base_ball_w
        self.gamma = gamma
        self.clamp_max = clamp_max

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

        # --- hardness signal: detached sq err of the MoG MEAN (mixture mean) ---
        w_mix = log_w.exp().unsqueeze(-1)                       # [T, B*N, K, 1]
        mu_mix = (w_mix * mu).sum(dim=2)                        # [T, B*N, 2]
        sq_err = ((mu_mix - tgt) ** 2).sum(dim=-1).detach()     # [T, B*N]

        # reshape order is [b0n0, b0n1, ..., b0n10, b1n0, ...]; ball is entity 0 -> p % N == 0.
        sq_err_e = sq_err.reshape(T, B, N)                      # [T, B, N]
        ball_hard = sq_err_e[:, :, 0]                           # [T, B]  (ball only)

        # focal relative hardness, per-timestep mean-normalized so mean ball weight == base_ball_w
        eps = 1e-6
        rel = ball_hard / (ball_hard.mean(dim=1, keepdim=True) + eps)   # [T, B], mean 1
        raw = rel.clamp(min=0.0).pow(self.gamma).clamp(max=self.clamp_max)
        w_ball = self.base_ball_w * raw / (raw.mean(dim=1, keepdim=True) + eps)  # [T, B], mean == base_ball_w

        # assemble per-(t, B*N) entity weight: ball positions get w_ball, players get 1
        w_ent = torch.ones(T, B, N, device=pred.device)
        w_ent[:, :, 0] = w_ball
        w_ent = w_ent.reshape(T, B * N)                        # [T, B*N]

        t_weights = torch.linspace(1.0, 2.0, T, device=pred.device)
        total = sum(
            t_weights[t] * (w_ent[t] * nll[t]).sum() / w_ent[t].sum()
            for t in range(T)
        )
        return total / t_weights.sum()


def build_trainer_class(base_ball_w: float, gamma: float, save_stem: str,
                        clamp_max: float = 20.0, seed: int = 0):
    """Return a Trainer subclass of v46's Trainer using the tail-weighted ball loss."""

    class _TailBallTrainer(nba_v46.Trainer):
        def train(self):
            torch.manual_seed(seed)
            random.seed(seed)
            generator = torch.Generator('cpu').manual_seed(seed)

            train_sampler = NBASampler(self.batch_size, self.train_dataset.max_start,
                                       seed=seed, shuffle=True)
            val_sampler = NBASampler(self.batch_size, self.val_dataset.max_start, shuffle=False)
            train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  sampler=train_sampler, generator=generator)
            val_dl = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                sampler=val_sampler, generator=generator)

            model = NBAv46Model().to(self.device)
            train_loss_fn = TailBallMoGNLLLoss(base_ball_w=base_ball_w, gamma=gamma,
                                               clamp_max=clamp_max)
            val_loss_fn = MultiStepMSE()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            for epoch in range(self.epochs):
                print(f'Epoch {epoch + 1}/{self.epochs}  '
                      f'(base_ball_w={base_ball_w}, gamma={gamma})')
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

    return _TailBallTrainer


# Headline config: gamma=1.0, base_ball_w=8 (mean ball weight == v60_w8, redistributed to tail).
Trainer = build_trainer_class(base_ball_w=8.0, gamma=1.0, save_stem='model_v71')
