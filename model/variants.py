"""Ensemble members = the one `NBAv46Model`, trained differently.

Every member is built by `build_member(seed, ball_w)`:
  - `seed`    varies weight init + data order (decorrelated members average better)
  - `ball_w`  up-weights the ball's loss; ball_w == 1.0 is the plain generalist loss,
              ball_w > 1.0 makes a ball specialist.

The 6 final members (see README):
  player generalists : build_member(seed=0)            -> v46
                       build_member(seed=3)            -> v46_s3
                       build_member(seed=7)            -> v46_s7
  ball specialists   : build_member(seed=0, ball_w=8)  -> v60_w8
                       build_member(seed=1, ball_w=5)  -> v60_w5_s1
                       build_member(seed=2, ball_w=8)  -> v60_w8_s2
"""
import math

import torch

from .architecture import Trainer, MoGNLLLoss, K_MIX


class BallWeightedMoGNLLLoss:
    """MoG NLL, but each entity's NLL is scaled; the ball (entity 0) by `ball_w`."""

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
        nll = -log_mix

        # reshape order is [b0n0, b0n1, ..., b0n10, b1n0, ...]; ball is always entity index 0.
        w_ent = torch.ones(N, device=pred.device)
        w_ent[0] = self.ball_w
        w_ent = w_ent.repeat(B)  # [B*N]
        denom = w_ent.sum()

        t_weights = torch.linspace(1.0, 2.0, T, device=pred.device)
        total = sum(t_weights[t] * (w_ent * nll[t]).sum() / denom for t in range(T))
        return total / t_weights.sum()


def build_member(seed: int = 0, ball_w: float = 1.0):
    """Return a Trainer subclass for one ensemble member. Discovered by core/run_one.py,
    which names the saved checkpoint after the model file."""
    loss = MoGNLLLoss() if ball_w == 1.0 else BallWeightedMoGNLLLoss(ball_w)

    class _Member(Trainer):
        def __init__(self, *args, **kwargs):
            kwargs['seed'] = seed
            kwargs['train_loss_fn'] = loss
            super().__init__(*args, **kwargs)

    return _Member
