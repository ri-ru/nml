"""v61: HiVT-style multi-agent trajectory forecaster.

Departures from v46 (GRU + per-step MoG, autoregressive):
  - Non-autoregressive: predicts all 12 future steps in one shot. No rollout
    error accumulation, no mean-feedback.
  - Per-focal-agent canonical reference frame: each agent's view is rotated so
    its recent heading is along +x and translated so its last context position
    is the origin. Done in real feet (not normalized space) so anisotropic
    sigma does not distort rotations.
  - Hierarchical encoder:
      * Per-agent temporal Transformer over the agent's OWN canonical trajectory
      * L layers of agent-agent (AA) graph attention at the last context step,
        in each focal agent's frame, with distance + type-pair edge biases
        (carried over from v46.TypeAwareDistanceBiasedAttention).
  - Trajectory-level mixture output: K=6 learned mode queries decode entire
    12-step futures (mu_x, mu_y, log_sigma in canonical frame) + K mode logits
    per agent. Isotropic Gaussian (scalar sigma) since rotation-invariant.
  - Loss: WTA NLL + classification CE, with a mixture-NLL warmup over the first
    ~10% of epochs to prevent early mode collapse.

Dataset / Sampler / Trainer interface match v46 so `run_one.py` works unchanged:
    python run_one.py models/nba_v61.py 200 32
"""
import math
import os
import random
from pathlib import Path
from pathlib import PureWindowsPath

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
K_MODES = 6
SPEED_EPS = 1e-2  # feet/frame: below this, fall back to identity rotation


# ----------------------------------------------------------------------------
# Dataset (identical to v46)
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
# Canonical-frame helper
# ----------------------------------------------------------------------------

def build_canonical_transform(pos_real):
    """Build per-(B, N_focal) rotation that maps GLOBAL vectors -> focal CANONICAL.

    canonical_vec = R @ global_vec  (so focal agent's heading lies along +x).

    Args:
        pos_real: [B, Tc, N, 2] in real feet.
    Returns:
        R:           [B, N, 2, 2]
        p_last_real: [B, N, 2]   last-context position in real feet.
    """
    B, Tc, N, _ = pos_real.shape
    p_last_real = pos_real[:, -1]                                   # [B, N, 2]
    n_avg = min(3, Tc - 1)
    vel = pos_real[:, -n_avg:] - pos_real[:, -n_avg - 1:-1]         # [B, n_avg, N, 2]
    vel_avg = vel.mean(dim=1)                                       # [B, N, 2]
    speed = vel_avg.norm(dim=-1, keepdim=True)                      # [B, N, 1]
    fallback = torch.zeros_like(vel_avg)
    fallback[..., 0] = 1.0
    safe = speed > SPEED_EPS
    heading = torch.where(safe, vel_avg / speed.clamp(min=SPEED_EPS), fallback)
    cos_t = heading[..., 0]
    sin_t = heading[..., 1]
    R = torch.stack([
        torch.stack([cos_t,  sin_t], dim=-1),
        torch.stack([-sin_t, cos_t], dim=-1),
    ], dim=-2)                                                       # [B, N, 2, 2]
    return R, p_last_real


# ----------------------------------------------------------------------------
# Submodules
# ----------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Per-agent temporal Transformer over the agent's OWN canonical (pos, vel)."""

    def __init__(self, D=128, num_heads=4, num_layers=2, max_len=16):
        super().__init__()
        self.D = D
        self.in_proj = nn.Linear(4, D)
        self.pos_emb = nn.Parameter(torch.zeros(max_len, D))
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads, dim_feedforward=D * 4,
            batch_first=True, activation='gelu', norm_first=True, dropout=0.0,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(D))

    def forward(self, traj_self):
        """traj_self: [BN, Tc, 4]  -> [BN, D]"""
        BN, Tc, _ = traj_self.shape
        x = self.in_proj(traj_self) + self.pos_emb[:Tc].unsqueeze(0)
        cls = self.cls.expand(BN, 1, self.D)
        x = torch.cat([cls, x], dim=1)
        x = self.tr(x)
        return x[:, 0]


class AAGraphAttention(nn.Module):
    """Agent-Agent graph attention at the last context step, in focal frame.

    Carries over the distance + type-pair edge biases from v46. Pre-norm with
    a small FFN; one layer is essentially a v46-style block plus FFN.
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
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(
            nn.Linear(D, D * 4), nn.GELU(), nn.Linear(D * 4, D),
        )
        self.dist_proj = nn.Linear(1, num_heads)
        self.type_pair_bias = nn.Embedding(num_types * num_types, num_heads)

    def forward(self, tokens, canon_pos_last, node_type):
        """tokens:         [BNf, N, D]
        canon_pos_last: [BNf, N, 2]   positions of N entities in focal's frame at t = Tc-1
        node_type:      [BNf, N] long
        Returns:        [BNf, N, D]
        """
        BNf, N, D = tokens.shape
        h = self.norm1(tokens)
        diff = canon_pos_last.unsqueeze(2) - canon_pos_last.unsqueeze(1)   # [BNf, N, N, 2]
        dist = diff.norm(dim=-1, keepdim=True)                              # [BNf, N, N, 1]
        dist_bias = self.dist_proj(dist).permute(0, 3, 1, 2)                # [BNf, H, N, N]
        type_i = node_type.unsqueeze(2).expand(-1, -1, N)
        type_j = node_type.unsqueeze(1).expand(-1, N, -1)
        type_bias = self.type_pair_bias(type_i * 3 + type_j).permute(0, 3, 1, 2)

        def split_heads(t):
            return t.reshape(BNf, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q(h))
        K = split_heads(self.k(h))
        V = split_heads(self.v(h))
        attn = F.softmax(Q @ K.transpose(-2, -1) * self.scale + dist_bias + type_bias, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(BNf, N, D)
        h = tokens + self.out(out)
        h = h + self.ff(self.norm2(h))
        return h


class ModeDecoder(nn.Module):
    """K trajectory queries -> K full (T_h x 3) futures per focal agent.

    Output per (focal, mode, t): (mu_x_canon, mu_y_canon, log_sigma).
    Plus K mode logits per focal.
    """

    def __init__(self, D=128, K=K_MODES, T_h=12):
        super().__init__()
        self.K = K
        self.T_h = T_h
        self.mode_query = nn.Parameter(torch.randn(K, D) * 0.02)
        self.traj_head = nn.Sequential(
            nn.Linear(2 * D, D * 2), nn.GELU(),
            nn.Linear(D * 2, D * 2), nn.GELU(),
            nn.Linear(D * 2, T_h * 3),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(),
            nn.Linear(D, K),
        )

    def forward(self, h_focal):
        """h_focal: [B, N, D]
        Returns:
            mu_canon:  [B, N, K, T_h, 2]
            log_sigma: [B, N, K, T_h]
            logits:    [B, N, K]
        """
        B, N, D = h_focal.shape
        h_tiled = h_focal.unsqueeze(2).expand(B, N, self.K, D)
        q_tiled = self.mode_query.expand(B, N, self.K, D)
        x = torch.cat([h_tiled, q_tiled], dim=-1)
        out = self.traj_head(x).reshape(B, N, self.K, self.T_h, 3)
        mu_canon = out[..., :2]
        log_sigma = out[..., 2].clamp(-3, 3)
        logits = self.cls_head(h_focal)
        return mu_canon, log_sigma, logits


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class NBAv61Model(nn.Module):
    def __init__(self, mu, sigma, state_dim=128, K=K_MODES, num_heads=4,
                 num_temporal_layers=2, num_aa_layers=2,
                 context_size=8, horizon_size=12):
        super().__init__()
        self.state_dim = state_dim
        self.K = K
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.register_buffer('mu_g',    torch.as_tensor(mu,    dtype=torch.float))
        self.register_buffer('sigma_g', torch.as_tensor(sigma, dtype=torch.float))

        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        # AA token input: canon_pos (2) + canon_vel (2) + team_emb (TEAM_EMB_DIM)
        self.token_in = nn.Linear(4 + TEAM_EMB_DIM, state_dim)

        self.temporal_enc = TemporalEncoder(
            D=state_dim, num_heads=num_heads,
            num_layers=num_temporal_layers, max_len=context_size + 1,
        )
        self.aa_layers = nn.ModuleList([
            AAGraphAttention(D=state_dim, num_heads=num_heads)
            for _ in range(num_aa_layers)
        ])
        self.fuse = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim), nn.GELU(),
            nn.LayerNorm(state_dim),
        )
        self.decoder = ModeDecoder(D=state_dim, K=K, T_h=horizon_size)

    def _node_type(self, X):
        team = X[:, 0, :, 3]
        nt = torch.zeros_like(team, dtype=torch.long)
        nt[team == -1] = 1
        nt[team == 1] = 2
        return nt

    def forward(self, X):
        """X: [B, Tc, N, 4]  (spatial dims normalized by global (mu, sigma)).

        Training: returns dict (predictions in canonical frame + transform metadata).
        Eval:     returns [T_h, B*N, 2] in the global-normalized frame.
        """
        B, Tc, N, _ = X.shape
        K = self.K
        Th = self.horizon_size
        D = self.state_dim
        device = X.device
        mu_g = self.mu_g.to(device)
        sigma_g = self.sigma_g.to(device)

        # 1. Real-coord positions and velocities
        pos_real = X[..., :2] * sigma_g + mu_g                   # [B, Tc, N, 2]
        vel_real = torch.zeros_like(pos_real)
        vel_real[:, 1:] = pos_real[:, 1:] - pos_real[:, :-1]

        # 2. Per-focal-agent canonical transform
        R, p_last_real = build_canonical_transform(pos_real)
        node_type = self._node_type(X)                            # [B, N]

        # 3. Canonical positions / velocities of every entity in every focal frame
        #    canon[b, t, nf, no, :] = R[b, nf] @ (pos_real[b, t, no] - p_last_real[b, nf])
        diff = pos_real.unsqueeze(2) - p_last_real.unsqueeze(1).unsqueeze(3)  # [B,Tc,N,N,2]
        R_b = R.unsqueeze(1).unsqueeze(3)                                      # [B,1,N,1,2,2]
        cx = R_b[..., 0, 0] * diff[..., 0] + R_b[..., 0, 1] * diff[..., 1]
        cy = R_b[..., 1, 0] * diff[..., 0] + R_b[..., 1, 1] * diff[..., 1]
        canon_pos = torch.stack([cx, cy], dim=-1)                              # [B,Tc,N,N,2]
        vd = vel_real.unsqueeze(2)                                             # [B,Tc,1,N,2]
        vx = R_b[..., 0, 0] * vd[..., 0] + R_b[..., 0, 1] * vd[..., 1]
        vy = R_b[..., 1, 0] * vd[..., 0] + R_b[..., 1, 1] * vd[..., 1]
        canon_vel = torch.stack([vx, vy], dim=-1)                              # [B,Tc,N,N,2]

        # 4. Temporal encoding: focal nf encodes its OWN trajectory in its OWN frame.
        idx = torch.arange(N, device=device)
        focal_self_pos = canon_pos[:, :, idx, idx]                             # [B,Tc,N,2]
        focal_self_vel = canon_vel[:, :, idx, idx]                             # [B,Tc,N,2]
        traj_self = torch.cat([focal_self_pos, focal_self_vel], dim=-1)        # [B,Tc,N,4]
        traj_self = traj_self.permute(0, 2, 1, 3).reshape(B * N, Tc, 4)
        h_temporal = self.temporal_enc(traj_self).reshape(B, N, D)             # [B,N,D]

        # 5. AA graph attention at the last context step, in each focal's frame.
        canon_pos_last = canon_pos[:, -1]                                      # [B,N,N,2]
        canon_vel_last = canon_vel[:, -1]                                      # [B,N,N,2]
        team_emb = self.team_emb(node_type)                                    # [B,N,TEAM]
        team_emb_o = team_emb.unsqueeze(1).expand(B, N, N, TEAM_EMB_DIM)       # other-agent team_emb
        token_in = torch.cat([canon_pos_last, canon_vel_last, team_emb_o], dim=-1)
        tokens = self.token_in(token_in)                                       # [B,N,N,D]

        # Inject each OTHER agent's temporal-encoded "motion character" into its token.
        h_temporal_o = h_temporal.unsqueeze(1).expand(B, N, N, D)
        tokens = tokens + h_temporal_o

        BNf = B * N
        tokens_flat = tokens.reshape(BNf, N, D)
        canon_pos_last_flat = canon_pos_last.reshape(BNf, N, 2)
        node_type_o = node_type.unsqueeze(1).expand(B, N, N).reshape(BNf, N)
        for layer in self.aa_layers:
            tokens_flat = layer(tokens_flat, canon_pos_last_flat, node_type_o)
        tokens = tokens_flat.reshape(B, N, N, D)
        h_focal_aa = tokens[:, idx, idx]                                       # [B,N,D]

        # 6. Fuse focal temporal + focal AA tokens
        h_focal = self.fuse(torch.cat([h_temporal, h_focal_aa], dim=-1))       # [B,N,D]

        # 7. Mode decoder
        mu_canon, log_sigma, logits = self.decoder(h_focal)

        if self.training:
            return {
                'mu_canon':    mu_canon,
                'log_sigma':   log_sigma,
                'logits':      logits,
                'R':           R,
                'p_last_real': p_last_real,
                'mu_g':        mu_g,
                'sigma_g':     sigma_g,
            }

        # 8. Inference: mixture mean per focal, inverse-rotate, re-normalize.
        probs = F.softmax(logits, dim=-1)                                      # [B,N,K]
        mu_mix_canon = (probs.unsqueeze(-1).unsqueeze(-1) * mu_canon).sum(dim=2)  # [B,N,Th,2]
        Rt = R.transpose(-1, -2)                                               # R is orthogonal
        Rt_b = Rt.unsqueeze(2)                                                 # [B,N,1,2,2]
        dx_c = mu_mix_canon[..., 0]
        dy_c = mu_mix_canon[..., 1]
        dx_r = Rt_b[..., 0, 0] * dx_c + Rt_b[..., 0, 1] * dy_c
        dy_r = Rt_b[..., 1, 0] * dx_c + Rt_b[..., 1, 1] * dy_c
        mu_real = torch.stack([dx_r, dy_r], dim=-1) + p_last_real.unsqueeze(2)  # [B,N,Th,2]
        mu_norm = (mu_real - mu_g) / sigma_g                                    # [B,N,Th,2]
        return mu_norm.permute(2, 0, 1, 3).reshape(Th, B * N, 2)


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------

class WTAMoGLoss:
    """Winner-Takes-All NLL on the closest mode + classification CE on mode logits.

    A linear warmup blends in full-mixture NLL over the first
    `mixture_warmup_epochs` epochs to keep modes diverse early on.
    """

    def __init__(self, mixture_warmup_epochs=20, ce_weight=0.5):
        self.mixture_warmup_epochs = mixture_warmup_epochs
        self.ce_weight = ce_weight
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def compute(self, pred, target):
        mu_canon    = pred['mu_canon']     # [B, N, K, Th, 2]
        log_sigma   = pred['log_sigma']    # [B, N, K, Th]
        logits      = pred['logits']       # [B, N, K]
        R           = pred['R']            # [B, N, 2, 2]
        p_last_real = pred['p_last_real'] # [B, N, 2]
        mu_g        = pred['mu_g']
        sigma_g     = pred['sigma_g']

        B, N, K, Th, _ = mu_canon.shape
        device = mu_canon.device

        # Target -> each focal's canonical frame
        tgt_norm = target[..., :2].permute(0, 2, 1, 3)                          # [B,N,Th,2]
        tgt_real = tgt_norm * sigma_g + mu_g
        d_real = tgt_real - p_last_real.unsqueeze(2)                            # [B,N,Th,2]
        Rb = R.unsqueeze(2)                                                      # [B,N,1,2,2]
        tx_c = Rb[..., 0, 0] * d_real[..., 0] + Rb[..., 0, 1] * d_real[..., 1]
        ty_c = Rb[..., 1, 0] * d_real[..., 0] + Rb[..., 1, 1] * d_real[..., 1]
        tgt_canon = torch.stack([tx_c, ty_c], dim=-1)                            # [B,N,Th,2]

        diff = mu_canon - tgt_canon.unsqueeze(2)                                 # [B,N,K,Th,2]
        sq = (diff ** 2).sum(dim=-1)                                             # [B,N,K,Th]
        tw = torch.linspace(1.0, 2.0, Th, device=device)
        per_mode_err = (sq * tw).sum(dim=-1) / tw.sum()                          # [B,N,K]
        with torch.no_grad():
            best = per_mode_err.argmin(dim=-1)                                   # [B,N]

        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        n_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        mu_best = mu_canon[b_idx, n_idx, best]                                   # [B,N,Th,2]
        ls_best = log_sigma[b_idx, n_idx, best]                                  # [B,N,Th]
        var_best = (2 * ls_best).exp()
        sq_best = ((tgt_canon - mu_best) ** 2).sum(dim=-1)                       # [B,N,Th]
        # Isotropic 2D Gaussian NLL: -log p = 0.5 ||y-mu||^2/sigma^2 + 2 log sigma + log(2pi)
        nll = 0.5 * sq_best / var_best + 2 * ls_best + math.log(2 * math.pi)
        nll_loss = ((nll * tw).sum(dim=-1) / tw.sum()).mean()

        ce = F.cross_entropy(logits.reshape(B * N, K), best.reshape(B * N))

        if self.epoch < self.mixture_warmup_epochs:
            var = (2 * log_sigma).exp()                                          # [B,N,K,Th]
            log_prob = -0.5 * sq / var - 2 * log_sigma - math.log(2 * math.pi)   # [B,N,K,Th]
            log_w = logits - logits.logsumexp(dim=-1, keepdim=True)              # [B,N,K]
            log_mix = torch.logsumexp(log_w.unsqueeze(-1) + log_prob, dim=2)     # [B,N,Th]
            mix_loss = (-(log_mix * tw).sum(dim=-1) / tw.sum()).mean()
            alpha = 1.0 - self.epoch / self.mixture_warmup_epochs
            return alpha * mix_loss + (1 - alpha) * (nll_loss + self.ce_weight * ce)
        return nll_loss + self.ce_weight * ce


class MultiStepMSE:
    """Val-time MSE in normalized space (same shape contract as v46)."""

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
    def __init__(self, train_path, val_path, device='cuda', epochs=200, batch_size=32, **kwargs):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

        if TRAIN_TXT.exists() and VAL_TXT.exists():
            train_files = TRAIN_TXT.read_text().splitlines()
            val_files = VAL_TXT.read_text().splitlines()
        else:
            train_files = [str(p) for p in Path(train_path).glob('*.pt')]
            val_files = [str(p) for p in Path(val_path).glob('*.pt')]

        # If the path lists were authored on a different machine (e.g. Windows
        # paths committed by a teammate), rebase basenames against NML_DATA_DIR
        # — or against `train_path` if NML_DATA_DIR is not set and train_path
        # actually exists locally.
        data_dir = os.environ.get('NML_DATA_DIR')
        if data_dir is None and Path(train_path).exists():
            data_dir = train_path
        if data_dir is not None:
            data_dir = Path(data_dir)

            def _rebase(p):
                if Path(p).exists():
                    return p
                # Handle both POSIX and Windows-style paths.
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

        model = NBAv61Model(mu=self.mu, sigma=self.sigma).to(self.device)
        train_loss_fn = WTAMoGLoss(mixture_warmup_epochs=max(10, self.epochs // 10))
        val_loss_fn   = MultiStepMSE()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss_fn.set_epoch(epoch)
            model.train()
            self._run_epoch(train_dl, train_loss_fn, model, optimizer, 'train')
            model.train(False)
            with torch.no_grad():
                self._run_epoch(val_dl, val_loss_fn, model, None, 'val')
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            scheduler.step()

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v61.pth')
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
