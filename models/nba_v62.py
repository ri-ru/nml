"""v62: v46's GRU-AR + MoG-K2 backbone, with v61's per-focal-agent canonical frame.

The hypothesis from the v61 ensemble analysis (v61 received 0.158 global weight
and 0.204 weight on the ball alone) is that per-focal-agent canonical frames are
genuinely useful — particularly for the ball. v61's other choices (non-AR,
mode-query decoder) however cost it elsewhere. v62 keeps v61's canonical-frame
preprocessing but reverts everything else to v46:
  - GRU autoregressive rollout (no error-accumulation problem in 12 steps).
  - Per-step MoG K=2 output (proven on the 21-model ensemble).
  - v46-style social features (K=2 nearest same/opp + soft-attention pooled)
    computed IN EACH FOCAL'S CANONICAL FRAME.
  - v46-style AA attention with distance + type-pair edge biases, but the
    attention happens per-focal-agent (one self-attention pass over N tokens
    per focal, batched as B*N_focal — same trick as v61).
  - Loss: time-weighted MoG NLL with targets converted to each focal's canonical
    frame (same NLL formula as v46, applied after the canonical rotation).

Eval-mode output is [Th, B*N, 2] in the global-normalized frame, so the
ensemble pipeline (save_*_tta_preds.py, weighted_ensemble.py) works unchanged.

Run: python run_one.py models/nba_v62.py 300 32
"""
import math
import os
import random
from pathlib import Path, PureWindowsPath

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

_DIR = Path(__file__).parent
WEIGHTS_DIR = _DIR / 'weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_TXT = _DIR.parent / 'train_files.txt'
VAL_TXT   = _DIR.parent / 'val_files.txt'

TEAM_EMB_DIM = 8
K_MIX = 2
SPEED_EPS = 1e-2  # ft/frame — below this we fall back to identity rotation
INF = 1e4         # finite "infinity" for masked softmax (same value as v46)


# ----------------------------------------------------------------------------
# Dataset / Sampler — identical to v46 / v61
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Canonical-frame helper (identical contract to v61)
# ----------------------------------------------------------------------------

def build_canonical_transform(pos_real_ctx):
    """Build per-(B, N_focal) rotation that maps GLOBAL vectors -> focal CANONICAL.

    pos_real_ctx: [B, Tc, N, 2] in real feet (context only).
    Returns:
        R:           [B, N, 2, 2]   canonical = R @ global
        p_last_real: [B, N, 2]      focal's last-context position in real feet
    """
    B, Tc, N, _ = pos_real_ctx.shape
    p_last_real = pos_real_ctx[:, -1]
    n_avg = min(3, Tc - 1)
    vel = pos_real_ctx[:, -n_avg:] - pos_real_ctx[:, -n_avg - 1:-1]  # [B, n_avg, N, 2]
    vel_avg = vel.mean(dim=1)                                         # [B, N, 2]
    speed = vel_avg.norm(dim=-1, keepdim=True)                        # [B, N, 1]
    fallback = torch.zeros_like(vel_avg)
    fallback[..., 0] = 1.0
    safe = speed > SPEED_EPS
    heading = torch.where(safe, vel_avg / speed.clamp(min=SPEED_EPS), fallback)
    cos_t = heading[..., 0]
    sin_t = heading[..., 1]
    R = torch.stack([
        torch.stack([cos_t,  sin_t], dim=-1),
        torch.stack([-sin_t, cos_t], dim=-1),
    ], dim=-2)
    return R, p_last_real


# ----------------------------------------------------------------------------
# Submodules — GRU cell (from v46) + per-focal AA attention (from v61)
# ----------------------------------------------------------------------------

class GRUCell(nn.Module):
    """v46's GRU cell — pre-LayerNorm + standard Z/R/H gates."""

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


class PerFocalAttention(nn.Module):
    """Self-attention over N entity tokens in each focal's canonical frame.

    Same structure as v61.AAGraphAttention but called once per timestep during
    the AR rollout. Distance bias is computed from canonical positions; since
    Euclidean distance is rotation-invariant, this equals the global pairwise
    distance. Type-pair bias is frame-independent.
    """

    def __init__(self, D=128, num_heads=4, num_types=3):
        super().__init__()
        assert D % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = D // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(D, D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.out = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
        self.dist_proj = nn.Linear(1, num_heads)
        self.type_pair_bias = nn.Embedding(num_types * num_types, num_heads)

    def forward(self, tokens, canon_pos, node_type):
        """tokens:    [BNf, N, D]  per-focal-agent token bag of N entities
        canon_pos: [BNf, N, 2]  positions of N entities in focal's frame
        node_type: [BNf, N] long
        Returns:   [BNf, N, D]
        """
        BNf, N, D = tokens.shape
        diff = canon_pos.unsqueeze(2) - canon_pos.unsqueeze(1)            # [BNf, N, N, 2]
        dist = diff.norm(dim=-1, keepdim=True)                             # [BNf, N, N, 1]
        dist_bias = self.dist_proj(dist).permute(0, 3, 1, 2)               # [BNf, H, N, N]
        type_i = node_type.unsqueeze(2).expand(-1, -1, N)
        type_j = node_type.unsqueeze(1).expand(-1, N, -1)
        type_bias = self.type_pair_bias(type_i * 3 + type_j).permute(0, 3, 1, 2)

        def split_heads(t):
            return t.reshape(BNf, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q(tokens))
        K = split_heads(self.k(tokens))
        V = split_heads(self.v(tokens))
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale + dist_bias + type_bias, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(BNf, N, D)
        return self.norm(tokens + self.out(out))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class NBAv62Model(nn.Module):
    """Per-focal-agent canonical frame + v46 GRU/attention/MoG, autoregressive.

    State per focal:
      - Hidden state h_f (per layer)
      - Predicted position fed back through R^T transform into global real coords

    Output (train): dict with raw MoG outputs in canonical frame + transform metadata.
    Output (eval):  [Th, B*N, 2] in global-normalized frame (drop-in to ensemble pipeline).
    """

    def __init__(self, mu, sigma, state_dim=128, K=K_MIX, num_heads=4, num_layers=2,
                 context_size=8, horizon_size=12):
        super().__init__()
        self.state_dim = state_dim
        self.K = K
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.num_layers = num_layers

        # Real <-> normalized conversion baked into the model
        self.register_buffer('mu_g',    torch.as_tensor(mu,    dtype=torch.float))
        self.register_buffer('sigma_g', torch.as_tensor(sigma, dtype=torch.float))

        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        # Per-focal scalar input feature dim:
        #   motion (8) = focal pos(2) + focal vel(2) + offset to ball(2) + vel offset to ball(2)
        #   social (24) = K=2 NN same/opp rel pos+vel (16) + soft-att same/opp pos+vel (8)
        #   team emb (TEAM_EMB_DIM) = 8
        # Total per-focal input: 40, same as v46.
        input_dim_0 = 32 + TEAM_EMB_DIM

        # Per-focal entity token features for AA attention:
        #   canon_pos(2) + canon_vel(2) + team_emb of OTHER agent (TEAM_EMB_DIM)
        self.entity_token_in = nn.Linear(4 + TEAM_EMB_DIM, state_dim)

        self.log_temperature = nn.Parameter(torch.zeros(1))

        self.gru_layers = nn.ModuleList([
            GRUCell(input_dim_0 if i == 0 else state_dim, state_dim)
            for i in range(num_layers)
        ])
        self.attn_layers = nn.ModuleList([
            PerFocalAttention(D=state_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        # MoG K head — identical to v46
        self.proj = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, K * 5),
        )

    def _node_type(self, X):
        team = X[:, 0, :, 3]
        nt = torch.zeros_like(team, dtype=torch.long)
        nt[team == -1] = 1
        nt[team == 1] = 2
        return nt

    def _per_focal_features(self, pos_real_now, vel_real, R, p_last_real,
                            node_type, team_emb, ball_mask):
        """Build per-focal canonical features for one timestep.

        Returns:
            focal_input: [B*N, 40]   per-focal GRU input
            entity_tokens: [B, N_focal, N_other, D]  per-focal entity tokens for AA
            canon_pos_focal_view: [B, N_focal, N_other, 2]  for distance bias
        """
        B, N, _ = pos_real_now.shape
        device = pos_real_now.device

        # diff[b, focal, other, :] = pos_real_now[b, other] - p_last_real[b, focal]
        diff = pos_real_now.unsqueeze(1) - p_last_real.unsqueeze(2)        # [B, N_f, N_o, 2]
        R_b = R.unsqueeze(2)                                                # [B, N_f, 1, 2, 2]
        canon_x = R_b[..., 0, 0] * diff[..., 0] + R_b[..., 0, 1] * diff[..., 1]
        canon_y = R_b[..., 1, 0] * diff[..., 0] + R_b[..., 1, 1] * diff[..., 1]
        canon_pos = torch.stack([canon_x, canon_y], dim=-1)                 # [B, N_f, N_o, 2]

        vd = vel_real.unsqueeze(1)                                          # [B, 1, N_o, 2]
        vx = R_b[..., 0, 0] * vd[..., 0] + R_b[..., 0, 1] * vd[..., 1]
        vy = R_b[..., 1, 0] * vd[..., 0] + R_b[..., 1, 1] * vd[..., 1]
        canon_vel = torch.stack([vx, vy], dim=-1)                           # [B, N_f, N_o, 2]

        # Focal's own canonical pos/vel (diagonal)
        idx = torch.arange(N, device=device)
        focal_self_pos = canon_pos[:, idx, idx]                             # [B, N, 2]
        focal_self_vel = canon_vel[:, idx, idx]                             # [B, N, 2]

        # Ball in each focal's frame: select via ball_mask, broadcast over focal dim
        ball_mask_o = ball_mask.float().unsqueeze(1).unsqueeze(-1)          # [B, 1, N_o, 1]
        ball_canon_pos = (canon_pos * ball_mask_o).sum(dim=2)               # [B, N_f, 2]
        ball_canon_vel = (canon_vel * ball_mask_o).sum(dim=2)               # [B, N_f, 2]

        # Motion features (8)
        motion = torch.cat([
            focal_self_pos,
            focal_self_vel,
            focal_self_pos - ball_canon_pos,
            focal_self_vel - ball_canon_vel,
        ], dim=-1)                                                          # [B, N, 8]

        # Social features (24)
        social = self._social_features(canon_pos, canon_vel, node_type, ball_mask)

        x_in = torch.cat([motion, social, team_emb], dim=-1)                # [B, N, 40]
        focal_input = x_in.reshape(B * N, -1)

        # Entity tokens for AA (per-focal)
        team_emb_o = team_emb.unsqueeze(1).expand(B, N, N, TEAM_EMB_DIM)
        entity_token_raw = torch.cat([canon_pos, canon_vel, team_emb_o], dim=-1)  # [B, N_f, N_o, 4+8]
        entity_tokens = self.entity_token_in(entity_token_raw)              # [B, N_f, N_o, D]
        return focal_input, entity_tokens, canon_pos

    def _social_features(self, canon_pos, canon_vel, node_type, ball_mask):
        """K=2 nearest same/opp + soft-att pooled, in each focal's frame.

        canon_pos: [B, N_focal, N_other, 2]   (focal at origin)
        canon_vel: [B, N_focal, N_other, 2]
        Returns:   [B, N_focal, 24]
        """
        B, N, _, _ = canon_pos.shape
        device = canon_pos.device

        # Distance from focal (origin) to entity j == ||canon_pos[focal, j]||
        # But for K-NN we want distances between PAIRS. Actually for v46-style
        # social features, the "nearest neighbor" is the focal's nearest same/opp.
        # Distance focal -> other = ||canon_pos[focal, other]||.
        dist_focal = canon_pos.norm(dim=-1)                                 # [B, N_f, N_o]

        type_focal = node_type.unsqueeze(2)                                 # [B, N_f, 1]
        type_other = node_type.unsqueeze(1)                                 # [B, 1, N_o]
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        same_mask = (type_focal == type_other) & (type_focal != 0) & ~eye
        opp_mask = ((type_focal == 1) & (type_other == 2)) | ((type_focal == 2) & (type_other == 1))

        dist_same = dist_focal.masked_fill(~same_mask, INF)
        dist_opp  = dist_focal.masked_fill(~opp_mask,  INF)

        # K=2 nearest indices
        nn_same_top2 = dist_same.topk(2, dim=2, largest=False).indices       # [B, N_f, 2]
        nn_opp_top2  = dist_opp.topk(2,  dim=2, largest=False).indices

        # Gather positions/velocities of those nearest entities (focal's view)
        b_e = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, 2)
        f_e = torch.arange(N, device=device).view(1, N, 1).expand(B, N, 2)
        rp_s = canon_pos[b_e, f_e, nn_same_top2]                            # [B, N, 2(K), 2(coord)]
        rv_s = canon_vel[b_e, f_e, nn_same_top2]
        rp_o = canon_pos[b_e, f_e, nn_opp_top2]
        rv_o = canon_vel[b_e, f_e, nn_opp_top2]

        # Zero out for ball focal (it has no teammates / opponents in this scheme)
        ball_focal_mask = ball_mask.unsqueeze(-1).unsqueeze(-1).expand(B, N, 2, 2)
        rp_s = rp_s.masked_fill(ball_focal_mask, 0.0)
        rv_s = rv_s.masked_fill(ball_focal_mask, 0.0)
        rp_o = rp_o.masked_fill(ball_focal_mask, 0.0)
        rv_o = rv_o.masked_fill(ball_focal_mask, 0.0)

        # Flatten K=2
        rp_s1 = rp_s[:, :, 0]; rp_s2 = rp_s[:, :, 1]
        rv_s1 = rv_s[:, :, 0]; rv_s2 = rv_s[:, :, 1]
        rp_o1 = rp_o[:, :, 0]; rp_o2 = rp_o[:, :, 1]
        rv_o1 = rv_o[:, :, 0]; rv_o2 = rv_o[:, :, 1]

        # Soft-attention pooled — same temperature scheme as v46
        T = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        w_same = torch.softmax(-dist_same / T, dim=2)                       # [B, N_f, N_o]
        w_opp  = torch.softmax(-dist_opp  / T, dim=2)
        att_same_pos = (w_same.unsqueeze(-1) * canon_pos).sum(dim=2)        # [B, N_f, 2]
        att_same_vel = (w_same.unsqueeze(-1) * canon_vel).sum(dim=2)
        att_opp_pos  = (w_opp.unsqueeze(-1)  * canon_pos).sum(dim=2)
        att_opp_vel  = (w_opp.unsqueeze(-1)  * canon_vel).sum(dim=2)

        ball2 = ball_mask.unsqueeze(-1).expand(B, N, 2)
        att_same_pos = att_same_pos.masked_fill(ball2, 0.0)
        att_same_vel = att_same_vel.masked_fill(ball2, 0.0)
        att_opp_pos  = att_opp_pos.masked_fill(ball2, 0.0)
        att_opp_vel  = att_opp_vel.masked_fill(ball2, 0.0)

        return torch.cat([
            rp_s1, rv_s1, rp_s2, rv_s2,
            rp_o1, rv_o1, rp_o2, rv_o2,
            att_same_pos, att_same_vel,
            att_opp_pos,  att_opp_vel,
        ], dim=-1)                                                          # [B, N, 24]

    def forward(self, X):
        """X: [B, Tc, N, 4] (spatial dims normalized by global (mu, sigma)).

        Train: returns dict (raw MoG outputs in canonical frame + transform).
        Eval:  returns [Th, B*N, 2] in global-normalized frame.
        """
        B, Tc, N, _ = X.shape
        K = self.K
        Th = self.horizon_size
        T = Tc + Th
        D = self.state_dim
        device = X.device
        mu_g = self.mu_g.to(device)
        sigma_g = self.sigma_g.to(device)

        # 1. Denormalize context
        pos_real_ctx = X[..., :2] * sigma_g + mu_g                          # [B, Tc, N, 2]
        ball_mask = (X[:, 0, :, 2] == 0)                                    # [B, N]
        node_type = self._node_type(X)                                      # [B, N]
        team_emb = self.team_emb(node_type)                                 # [B, N, TEAM_EMB_DIM]

        # 2. Canonical transform from context
        R, p_last_real = build_canonical_transform(pos_real_ctx)
        Rt = R.transpose(-1, -2)                                            # [B, N, 2, 2]
        idx = torch.arange(N, device=device)

        # 3. AR rollout
        h = [torch.zeros(B * N, D, device=device) for _ in range(self.num_layers)]
        all_preds = []
        prev_pos_real = None
        pos_real_now = None
        pred_mu_real = None

        for t in range(T):
            if t < Tc:
                pos_real_now = pos_real_ctx[:, t]                            # [B, N, 2]
            else:
                pos_real_now = pred_mu_real                                  # [B, N, 2]

            if prev_pos_real is None:
                vel_real = torch.zeros_like(pos_real_now)
            else:
                vel_real = pos_real_now - prev_pos_real
            prev_pos_real = pos_real_now

            # Per-focal canonical features
            focal_input, entity_tokens, canon_pos = self._per_focal_features(
                pos_real_now, vel_real, R, p_last_real,
                node_type, team_emb, ball_mask,
            )
            # focal_input: [B*N, 40]
            # entity_tokens: [B, N_f, N_o, D]
            # canon_pos: [B, N_f, N_o, 2]

            # GRU + AA layers (per focal)
            layer_in = focal_input
            for i in range(self.num_layers):
                h[i] = self.gru_layers[i](layer_in, h[i])

                # Inject focal's hidden state into the focal-self slot of its entity tokens.
                tokens = entity_tokens.clone()
                h_view = h[i].reshape(B, N, D)                              # [B, N_f, D]
                tokens[:, idx, idx, :] = tokens[:, idx, idx, :] + h_view

                # Batched self-attention: BNf = B * N_focal independent N-token sequences
                BNf = B * N
                tokens_flat = tokens.reshape(BNf, N, D)
                canon_pos_flat = canon_pos.reshape(BNf, N, 2)
                node_type_o = node_type.unsqueeze(1).expand(B, N, N).reshape(BNf, N)
                tokens_flat = self.attn_layers[i](tokens_flat, canon_pos_flat, node_type_o)
                # Take focal's own (now enriched) token as the new h
                tokens_out = tokens_flat.reshape(B, N, N, D)
                h[i] = tokens_out[:, idx, idx].reshape(BNf, D)

                layer_in = h[i]

            # 4. MoG K=2 output (only for the 12 horizon steps, indexed t = Tc-1 .. T-2)
            if Tc - 1 <= t < T - 1:
                out = self.proj(h[-1])                                       # [B*N, K*5]
                out_k = out.reshape(B * N, K, 5)
                log_w = out_k[:, :, 4] - out_k[:, :, 4].logsumexp(dim=-1, keepdim=True)
                w = log_w.exp()
                # Mixture mean in canonical frame, per focal
                pred_mu_canon = (w.unsqueeze(-1) * out_k[:, :, :2]).sum(dim=1)  # [B*N, 2]

                # Inverse-rotate + translate back to global real coords
                pmc = pred_mu_canon.reshape(B, N, 2)
                px = Rt[..., 0, 0] * pmc[..., 0] + Rt[..., 0, 1] * pmc[..., 1]
                py = Rt[..., 1, 0] * pmc[..., 0] + Rt[..., 1, 1] * pmc[..., 1]
                pred_mu_real = torch.stack([px, py], dim=-1) + p_last_real    # [B, N, 2]

                all_preds.append(out)

        preds = torch.stack(all_preds, dim=0)                                # [Th, B*N, K*5]

        if self.training:
            return {
                'preds':       preds,
                'R':           R,
                'p_last_real': p_last_real,
                'mu_g':        mu_g,
                'sigma_g':     sigma_g,
            }

        # 5. Eval: convert mixture-mean canonical predictions to global-normalized
        preds_k = preds.reshape(Th, B * N, K, 5)
        log_w = preds_k[:, :, :, 4] - preds_k[:, :, :, 4].logsumexp(dim=-1, keepdim=True)
        w = log_w.exp()
        pred_mu_canon_all = (w.unsqueeze(-1) * preds_k[:, :, :, :2]).sum(dim=2)  # [Th, B*N, 2]
        pmc = pred_mu_canon_all.reshape(Th, B, N, 2)
        # Inverse-rotate per focal
        Rt_b = Rt.unsqueeze(0)                                                # [1, B, N, 2, 2]
        px = Rt_b[..., 0, 0] * pmc[..., 0] + Rt_b[..., 0, 1] * pmc[..., 1]
        py = Rt_b[..., 1, 0] * pmc[..., 0] + Rt_b[..., 1, 1] * pmc[..., 1]
        pred_real = torch.stack([px, py], dim=-1) + p_last_real.unsqueeze(0)   # [Th, B, N, 2]
        pred_norm = (pred_real - mu_g) / sigma_g
        return pred_norm.reshape(Th, B * N, 2)


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------

class MoGCanonicalNLLLoss:
    """Time-weighted MoG-K NLL with targets transformed into each focal's canonical frame.

    Identical formula to v46's MoGNLLLoss; the only addition is the rotation of
    the target into the focal frame before the diff is taken.
    """

    def compute(self, pred, target):
        preds       = pred['preds']           # [Th, B*N, K*5]
        R           = pred['R']               # [B, N, 2, 2]
        p_last_real = pred['p_last_real']    # [B, N, 2]
        mu_g        = pred['mu_g']            # [2]
        sigma_g     = pred['sigma_g']         # [2]

        B, T, N, _ = target.shape             # T == Th
        Th = T
        K = K_MIX

        # Target in real coords -> per-focal canonical frame
        tgt_norm = target[..., :2].permute(1, 0, 2, 3)                       # [Th, B, N, 2]
        tgt_real = tgt_norm * sigma_g + mu_g                                  # [Th, B, N, 2]
        d_real = tgt_real - p_last_real.unsqueeze(0)                          # [Th, B, N, 2]
        Rb = R.unsqueeze(0)                                                   # [1, B, N, 2, 2]
        tx = Rb[..., 0, 0] * d_real[..., 0] + Rb[..., 0, 1] * d_real[..., 1]
        ty = Rb[..., 1, 0] * d_real[..., 0] + Rb[..., 1, 1] * d_real[..., 1]
        tgt_canon = torch.stack([tx, ty], dim=-1)                             # [Th, B, N, 2]
        tgt = tgt_canon.reshape(Th, B * N, 2)

        # MoG NLL (same form as v46.MoGNLLLoss)
        pk = preds.reshape(Th, B * N, K, 5)
        mu        = pk[:, :, :, :2]
        log_sigma = pk[:, :, :, 2:4].clamp(-3, 3)
        logit_w   = pk[:, :, :, 4]
        log_w = logit_w - logit_w.logsumexp(dim=-1, keepdim=True)
        var = (log_sigma * 2).exp()
        tgt_k = tgt.unsqueeze(2).expand_as(mu)
        diff = tgt_k - mu
        log_prob_k = -0.5 * (diff ** 2 / var + log_sigma * 2 + math.log(2 * math.pi))
        log_prob_k = log_prob_k.sum(dim=-1)
        log_mix = torch.logsumexp(log_w + log_prob_k, dim=-1)
        weights = torch.linspace(1.0, 2.0, Th, device=preds.device)
        total = sum(weights[t] * (-log_mix[t].mean()) for t in range(Th))
        return total / weights.sum()


class MultiStepMSE:
    """Val-time MSE in normalized space (same contract as v46)."""

    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def compute(self, pred, target):
        B, T, N, _ = target.shape
        target = target[..., :2].permute(1, 0, 2, 3).reshape(T, B * N, 2)
        return sum(self.loss_fn(pred[t], target[t]) for t in range(T)) / T


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------

class Trainer:
    def __init__(self, train_path, val_path, device='cuda', epochs=300, batch_size=32, **kwargs):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

        if TRAIN_TXT.exists() and VAL_TXT.exists():
            train_files = TRAIN_TXT.read_text().splitlines()
            val_files   = VAL_TXT.read_text().splitlines()
        else:
            train_files = [str(p) for p in Path(train_path).glob('*.pt')]
            val_files   = [str(p) for p in Path(val_path).glob('*.pt')]

        # Path rebase for cross-machine file lists (same trick as v61)
        data_dir = os.environ.get('NML_DATA_DIR')
        if data_dir is None and Path(train_path).exists():
            data_dir = train_path
        if data_dir is not None:
            data_dir = Path(data_dir)

            def _rebase(p):
                if Path(p).exists():
                    return p
                name = PureWindowsPath(p).name or Path(p).name
                return str(data_dir / name)

            train_files = [_rebase(p) for p in train_files]
            val_files   = [_rebase(p) for p in val_files]

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

        model = NBAv62Model(mu=self.mu, sigma=self.sigma).to(self.device)
        train_loss_fn = MoGCanonicalNLLLoss()
        val_loss_fn   = MultiStepMSE()
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

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v62.pth')
        self.model = model

    def _run_epoch(self, dataloader, loss_fn, model, optimizer, split):
        avg = 0.0
        for X, y in tqdm(dataloader, desc=f'  {split}'):
            X, y = X.to(self.device), y.to(self.device)
            pred = model(X)
            loss = loss_fn.compute(pred, y)
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            avg += loss.detach()
        avg = avg / len(dataloader)
        print(f'    {split} loss: {avg.item():.4f}')
