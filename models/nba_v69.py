"""v69: ball-only multi-mode MoG head (ball-specialist head with K_ball components).

Motivation (BALL_DEEP_ANALYSIS.md, section D): the model UNDER-MOVES the ball
(predicted net displacement 0.75x actual; 0.71x on big moves). This is mean-collapse:
the ball's future is genuinely multimodal (hold / pass-to-A / pass-to-B / shoot), but a
K=2 MoG cannot represent those discrete modes, so the mixture mean collapses toward "stay".

Lever: give the BALL (entity index 0, always) a DEDICATED output head with MORE mixture
components (K_ball in {4, 6}); players keep K=2. The shared GRU + attention trunk is
UNCHANGED from v46 — only the projection head is split + routed by entity type.

Design:
  * proj_player: Linear stack -> K_PLAYER*5   (mu_x, mu_y, log_sig_x, log_sig_y, logit_w)
  * proj_ball:   Linear stack -> K_BALL*5
  * Route entity 0 (ball) through proj_ball; entities 1..10 through proj_player.
  * Per-entity-K MoG NLL: ball NLL uses K_BALL components, player NLL uses K_PLAYER.
    Timestep weighting (linspace 1->2 over 12 steps) is preserved.
  * Autoregressive rollout: each entity's next-step pos = its own head's MoG weighted mean
    (ball -> ball head mean, players -> player head mean).
  * Eval returns the per-entity MoG WEIGHTED MEAN [T, B*N, 2] (MSE-optimal if modes are
    right; the point of more modes is to STOP the mean collapsing toward "stay").

Optional ball-weighting: BALL_W upweights the ball NLL in the loss (validated to help in
v60). v69.py keeps BALL_W=1.0 for a CLEAN A/B isolating the K_ball effect; thin variants
(nba_v69_k4.py / nba_v69_k6.py / nba_v69_k6_w4.py) set K_BALL / BALL_W and a save stem.

Default here: K_BALL=4, BALL_W=1.0.
"""
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_DIR = Path(__file__).parent
import sys
sys.path.insert(0, str(_DIR))
import nba_v46
from nba_v46 import (
    NBADataset, NBASampler, MultiStepMSE,
    GRUCell, TypeAwareDistanceBiasedAttention,
)

WEIGHTS_DIR = nba_v46.WEIGHTS_DIR
TEAM_EMB_DIM = nba_v46.TEAM_EMB_DIM

K_PLAYER = 2
K_BALL = 4          # overridden by thin variants
BALL_W = 1.0        # ball NLL upweight; 1.0 = clean A/B (no ball-weighting)


def _mog_mean(out_k):
    """out_k: [..., K, 5] -> weighted-mean [..., 2] over mixture components."""
    logit = out_k[..., 4]
    log_w = logit - logit.logsumexp(dim=-1, keepdim=True)
    w = log_w.exp().unsqueeze(-1)            # [..., K, 1]
    return (w * out_k[..., :2]).sum(dim=-2)  # [..., 2]


def _mog_dominant(out_k):
    """out_k: [..., K, 5] -> mu of the argmax-weight component [..., 2]."""
    logit = out_k[..., 4]                    # [..., K]
    idx = logit.argmax(dim=-1, keepdim=True) # [..., 1]
    idx2 = idx.unsqueeze(-1).expand(*idx.shape, 2)  # [..., 1, 2]
    return out_k[..., :2].gather(-2, idx2).squeeze(-2)


def _mog_nll(out_k, tgt, K):
    """out_k: [T, M, K*5] flattened; tgt: [T, M, 2]. Returns nll [T, M]."""
    T, M, _ = out_k.shape
    p = out_k.reshape(T, M, K, 5)
    mu = p[:, :, :, :2]
    log_sigma = p[:, :, :, 2:4].clamp(-3, 3)
    logit_w = p[:, :, :, 4]
    log_w = logit_w - logit_w.logsumexp(dim=-1, keepdim=True)
    var = (log_sigma * 2).exp()
    tgt_k = tgt.unsqueeze(2).expand_as(mu)
    diff = tgt_k - mu
    log_prob_k = -0.5 * (diff ** 2 / var + log_sigma * 2 + math.log(2 * math.pi))
    log_prob_k = log_prob_k.sum(dim=-1)               # [T, M, K]
    log_mix = torch.logsumexp(log_w + log_prob_k, dim=-1)  # [T, M]
    return -log_mix


class PerEntityKMoGNLLLoss:
    """Timestep-weighted MoG NLL where the ball uses K_BALL components and players use
    K_PLAYER. Optionally upweights the ball NLL by ball_w. The model returns a dict in
    training mode: {'ball': [T, B, K_BALL*5], 'player': [T, B*10, K_PLAYER*5]}."""

    def __init__(self, k_ball=K_BALL, k_player=K_PLAYER, ball_w=BALL_W):
        self.k_ball = k_ball
        self.k_player = k_player
        self.ball_w = ball_w

    def compute(self, pred, target):
        # target: [B, T, N, F]; ball is entity index 0.
        B, T, N, _ = target.shape
        tgt = target[:, :, :, :2].permute(1, 0, 2, 3)  # [T, B, N, 2]
        ball_tgt = tgt[:, :, 0, :].reshape(T, B, 2)            # [T, B, 2]
        player_tgt = tgt[:, :, 1:, :].reshape(T, B * (N - 1), 2)  # [T, B*10, 2]

        ball_nll = _mog_nll(pred['ball'], ball_tgt, self.k_ball)        # [T, B]
        player_nll = _mog_nll(pred['player'], player_tgt, self.k_player)  # [T, B*10]

        # Per-timestep mean over entities, matching v46 normalization: each entity slot
        # contributes equally, ball slot scaled by ball_w. v46/v60 average nll over all
        # B*N slots with the ball weighted ball_w; replicate that exactly.
        n_players = N - 1
        denom = self.ball_w + n_players  # per-batch-element entity-weight sum
        t_weights = torch.linspace(1.0, 2.0, T, device=pred['ball'].device)
        total = 0.0
        for t in range(T):
            # mean over players (B*10), mean over ball (B): combine as weighted sum / denom
            per_elem = (self.ball_w * ball_nll[t].mean()
                        + n_players * player_nll[t].mean()) / denom
            total = total + t_weights[t] * per_elem
        return total / t_weights.sum()


class NBAv69Model(nn.Module):
    """v46 trunk (GRU + type-aware attention + K=2 social features), with a SPLIT head:
    proj_ball (K_BALL mixture comps) for entity 0, proj_player (K_PLAYER) for entities 1..10."""

    def __init__(self, state_dim=128, context_size=8, horizon_size=12, num_heads=4,
                 num_layers=2, k_ball=K_BALL, k_player=K_PLAYER):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.k_ball = k_ball
        self.k_player = k_player

        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        input_dim_0 = 32 + TEAM_EMB_DIM
        self.log_temperature = nn.Parameter(torch.zeros(1))

        self.gru_layers = nn.ModuleList([
            GRUCell(input_dim_0 if i == 0 else state_dim, state_dim)
            for i in range(num_layers)
        ])
        self.attn_layers = nn.ModuleList([
            TypeAwareDistanceBiasedAttention(state_dim, num_heads)
            for _ in range(num_layers)
        ])
        # shared body of the head, then two type-specific output projections
        self.head_body = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.proj_player = nn.Linear(64, k_player * 5)
        self.proj_ball = nn.Linear(64, k_ball * 5)
        # how the ball mean is read out during rollout/eval: 'mean' (MSE-optimal) or 'mode'
        self.ball_readout = 'mean'

    # --- trunk helpers reused verbatim from v46 ---
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
        nn_same_top2 = dist_same.topk(2, dim=2, largest=False).indices
        nn_opp_top2 = dist_opp.topk(2, dim=2, largest=False).indices
        nn_same1, nn_same2 = nn_same_top2[:, :, 0], nn_same_top2[:, :, 1]
        nn_opp1, nn_opp2 = nn_opp_top2[:, :, 0], nn_opp_top2[:, :, 1]
        b_idx = torch.arange(B, device=pos.device).unsqueeze(1).expand(B, N)
        ball2 = ball_mask.unsqueeze(-1).expand(B, N, 2)

        def gather_rel(nn_idx):
            rp = (pos[b_idx, nn_idx] - pos).masked_fill(ball2, 0.0)
            rv = (vel[b_idx, nn_idx] - vel).masked_fill(ball2, 0.0)
            return rp, rv

        rp_s1, rv_s1 = gather_rel(nn_same1)
        rp_s2, rv_s2 = gather_rel(nn_same2)
        rp_o1, rv_o1 = gather_rel(nn_opp1)
        rp_o2, rv_o2 = gather_rel(nn_opp2)
        Tt = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        w_same = torch.softmax(-dist_same / Tt, dim=2)
        w_opp = torch.softmax(-dist_opp / Tt, dim=2)
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)
        att_same_pos = (w_same.unsqueeze(-1) * rel_pos).sum(dim=2).masked_fill(ball2, 0.0)
        att_same_vel = (w_same.unsqueeze(-1) * rel_vel).sum(dim=2).masked_fill(ball2, 0.0)
        att_opp_pos = (w_opp.unsqueeze(-1) * rel_pos).sum(dim=2).masked_fill(ball2, 0.0)
        att_opp_vel = (w_opp.unsqueeze(-1) * rel_vel).sum(dim=2).masked_fill(ball2, 0.0)
        return torch.cat([rp_s1, rv_s1, rp_s2, rv_s2,
                          rp_o1, rv_o1, rp_o2, rv_o2,
                          att_same_pos, att_same_vel,
                          att_opp_pos, att_opp_vel], dim=-1)

    def _heads(self, h_last, B, N):
        """h_last: [B*N, state_dim]. Returns:
        ball_out  [B, K_BALL*5]    (entity 0 of each batch element)
        player_out[B*(N-1), K_PLAYER*5]  (entities 1..N-1)
        and the per-entity mean [B*N, 2] for the rollout (ball mean uses ball head)."""
        body = self.head_body(h_last)              # [B*N, 64]
        body = body.reshape(B, N, 64)
        ball_body = body[:, 0, :]                  # [B, 64]
        player_body = body[:, 1:, :].reshape(B * (N - 1), 64)

        ball_out = self.proj_ball(ball_body)        # [B, K_BALL*5]
        player_out = self.proj_player(player_body)  # [B*(N-1), K_PLAYER*5]

        ball_k = ball_out.reshape(B, self.k_ball, 5)
        player_k = player_out.reshape(B, N - 1, self.k_player, 5)
        if self.ball_readout == 'mode':
            ball_mean = _mog_dominant(ball_k)              # [B, 2]
        else:
            ball_mean = _mog_mean(ball_k)                  # [B, 2]
        player_mean = _mog_mean(player_k)                  # [B, N-1, 2]

        mean = torch.empty(B, N, 2, device=h_last.device)
        mean[:, 0, :] = ball_mean
        mean[:, 1:, :] = player_mean
        return ball_out, player_out, mean.reshape(B * N, 2)

    def forward(self, X):
        B, _, N, _ = X.shape
        T = self.context_size + self.horizon_size
        ball_mask = (X[:, 0, :, 2] == 0)
        node_type = self._get_node_type(X)
        team_emb = self.team_emb(node_type)
        h = [torch.zeros(B * N, self.state_dim, device=X.device) for _ in range(self.num_layers)]
        ball_preds, player_preds = [], []
        prev_pos = None
        pred_mean = None

        for t in range(T):
            if t < self.context_size:
                pos = X[:, t, :, :2]
                vel = torch.zeros_like(pos) if prev_pos is None else pos - prev_pos
                prev_pos = pos
            else:
                pos = pred_mean.reshape(B, N, 2)
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
                ball_out, player_out, pred_mean = self._heads(h[-1], B, N)
                ball_preds.append(ball_out)       # [B, K_BALL*5]
                player_preds.append(player_out)   # [B*(N-1), K_PLAYER*5]

        ball_stack = torch.stack(ball_preds, dim=0)      # [T, B, K_BALL*5]
        player_stack = torch.stack(player_preds, dim=0)  # [T, B*(N-1), K_PLAYER*5]

        if self.training:
            return {'ball': ball_stack, 'player': player_stack}

        # eval: return per-entity weighted mean [T, B*N, 2]
        Th = ball_stack.shape[0]
        ball_k = ball_stack.reshape(Th, B, self.k_ball, 5)
        player_k = player_stack.reshape(Th, B, N - 1, self.k_player, 5)
        if self.ball_readout == 'mode':
            ball_mean = _mog_dominant(ball_k)          # [T, B, 2]
        else:
            ball_mean = _mog_mean(ball_k)              # [T, B, 2]
        player_mean = _mog_mean(player_k)              # [T, B, N-1, 2]
        out = torch.empty(Th, B, N, 2, device=X.device)
        out[:, :, 0, :] = ball_mean
        out[:, :, 1:, :] = player_mean
        return out.reshape(Th, B * N, 2)


def build_trainer_class(k_ball=K_BALL, ball_w=BALL_W, save_stem='model_v69', seed=0):
    """Trainer subclass using the split-head model + per-entity-K MoG NLL loss."""

    class _V69Trainer(nba_v46.Trainer):
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

            model = NBAv69Model(k_ball=k_ball).to(self.device)
            train_loss_fn = PerEntityKMoGNLLLoss(k_ball=k_ball, k_player=K_PLAYER, ball_w=ball_w)
            val_loss_fn = MultiStepMSE()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            for epoch in range(self.epochs):
                print(f'Epoch {epoch + 1}/{self.epochs}  (k_ball={k_ball}, ball_w={ball_w})')
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

    return _V69Trainer


# default Trainer for `run_one.py models\nba_v69.py` (K_BALL=4, BALL_W=1.0 clean A/B)
Trainer = build_trainer_class(k_ball=K_BALL, ball_w=BALL_W, save_stem='model_v69')
