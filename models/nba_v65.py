"""v65 — agent-centric ("HiVT-lite") model. The big structural bet (BALL_ERROR_PLAN / brainstorm).

Three ideas in one model:
  1. AGENT-CENTRIC FRAMES: each agent is predicted in its own local frame (translate to the agent's
     present position, rotate so its heading = +x). Makes the model inherently translation/rotation
     equivariant -- fixes the court-aligned brittleness in findings.md (rotation TTA scored 10.1).
  2. SPLIT ENCODE: per-agent temporal GRU encoder, then ONE global interaction-attention layer with
     local-frame relative geometry (HiVT's local-then-global structure).
  3. NON-AUTOREGRESSIVE DECODER: emit all 12 future MoG steps at once -> no rollout, no compounding
     drift (our error is dominated by late steps t=12).

Proven pieces kept: GRU cell (from v46), MoG K=2 head, timestep-weighted NLL. The MoG NLL is computed
in each agent's LOCAL frame (diagonal Gaussian valid there); eval transforms the predicted mean back
to global and reuses the standard MSE.
"""
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import nba_v46
from nba_v46 import GRUCell, NBASampler, MultiStepMSE, WEIGHTS_DIR

K_MIX = 2
TEAM_EMB_DIM = 8
CTX = 8
HOR = 12
EPS = 1e-3


# ---------- frame transforms (c, s = cos/sin of agent heading; o = origin) ----------
def make_frames(pos_ctx):
    """pos_ctx [B,CTX,N,2] -> c,s [B,N], o [B,N,2]. Heading from a 2-step present velocity."""
    o = pos_ctx[:, -1]                          # [B,N,2] present position
    hv = pos_ctx[:, -1] - pos_ctx[:, -3]        # [B,N,2]
    spd = hv.norm(dim=-1, keepdim=True)         # [B,N,1]
    unit = hv / spd.clamp(min=EPS)
    moving = (spd.squeeze(-1) > EPS)
    c = torch.where(moving, unit[..., 0], torch.ones_like(unit[..., 0]))
    s = torch.where(moving, unit[..., 1], torch.zeros_like(unit[..., 1]))
    return c, s, o


def pt_to_local(p, c, s, o):
    """Rotate global points p into local frames. p [...,2]; c,s,o broadcast to p's leading dims."""
    d = p - o
    lx = c * d[..., 0] + s * d[..., 1]
    ly = -s * d[..., 0] + c * d[..., 1]
    return torch.stack([lx, ly], dim=-1)


def vec_to_local(v, c, s):
    lx = c * v[..., 0] + s * v[..., 1]
    ly = -s * v[..., 0] + c * v[..., 1]
    return torch.stack([lx, ly], dim=-1)


def pt_to_global(lp, c, s, o):
    gx = c * lp[..., 0] - s * lp[..., 1]
    gy = s * lp[..., 0] + c * lp[..., 1]
    return torch.stack([gx, gy], dim=-1) + o


class AgentInteraction(nn.Module):
    """One multi-head attention over agents, biased by local-frame relative geometry + type pair."""
    def __init__(self, D, heads=4, num_types=3):
        super().__init__()
        self.h = heads
        self.hd = D // heads
        self.scale = self.hd ** -0.5
        self.q = nn.Linear(D, D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.o = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
        self.edge = nn.Linear(5, heads)                       # relpos(2)+relvel(2)+dist(1)
        self.type_bias = nn.Embedding(num_types * num_types, heads)

    def forward(self, z, relpos, relvel, dist, type_pair):
        B, N, D = z.shape
        ef = torch.cat([relpos, relvel, dist], dim=-1)        # [B,N,N,5]
        ebias = self.edge(ef).permute(0, 3, 1, 2)             # [B,h,N,N]
        tbias = self.type_bias(type_pair).permute(0, 3, 1, 2)  # [B,h,N,N]

        def sh(t):
            return t.reshape(B, N, self.h, self.hd).transpose(1, 2)

        Q, Kk, V = sh(self.q(z)), sh(self.k(z)), sh(self.v(z))
        attn = F.softmax(Q @ Kk.transpose(-2, -1) * self.scale + ebias + tbias, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.norm(z + self.o(out))


class NBAv65Model(nn.Module):
    def __init__(self, state_dim=128, num_heads=4, enc_layers=2):
        super().__init__()
        self.D = state_dim
        self.enc_layers = enc_layers
        self.team_emb = nn.Embedding(3, TEAM_EMB_DIM)
        self.enc_gru = nn.ModuleList([
            GRUCell(4 if i == 0 else state_dim, state_dim) for i in range(enc_layers)
        ])
        self.interact = AgentInteraction(state_dim, num_heads)
        self.dec = nn.Sequential(
            nn.Linear(state_dim + TEAM_EMB_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, HOR * K_MIX * 5),
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
        pos_ctx = X[:, :, :, :2]                       # [B,CTX,N,2]
        c, s, o = make_frames(pos_ctx)                 # c,s [B,N]; o [B,N,2]
        self._frames = (c, s, o)
        nt = self._node_type(X)
        team_emb = self.team_emb(nt)                   # [B,N,8]

        # --- per-agent local trajectory (origin at present, heading +x) ---
        c_t = c[:, None, :]; s_t = s[:, None, :]; o_t = o[:, None, :, :]
        traj_local = pt_to_local(pos_ctx, c_t, s_t, o_t)            # [B,CTX,N,2]
        vel_local = torch.zeros_like(traj_local)
        vel_local[:, 1:] = traj_local[:, 1:] - traj_local[:, :-1]
        inp = torch.cat([traj_local, vel_local], dim=-1)           # [B,CTX,N,4]

        # --- temporal GRU encoder ---
        h = [torch.zeros(B * N, self.D, device=X.device) for _ in range(self.enc_layers)]
        for t in range(CTX):
            layer_in = inp[:, t].reshape(B * N, 4)
            for i in range(self.enc_layers):
                h[i] = self.enc_gru[i](layer_in, h[i])
                layer_in = h[i]
        h_self = h[-1].reshape(B, N, self.D)

        # --- interaction in agent-i local frames ---
        rel = o.unsqueeze(1) - o.unsqueeze(2)                      # [B,N(i),N(j),2] global
        ci = c[:, :, None]; si = s[:, :, None]                     # frame of i
        relpos = torch.stack([ci * rel[..., 0] + si * rel[..., 1],
                              -si * rel[..., 0] + ci * rel[..., 1]], dim=-1)
        vel_pres = pos_ctx[:, -1] - pos_ctx[:, -2]                 # [B,N,2] present velocity
        rvel = vel_pres.unsqueeze(1) - vel_pres.unsqueeze(2)
        relvel = torch.stack([ci * rvel[..., 0] + si * rvel[..., 1],
                              -si * rvel[..., 0] + ci * rvel[..., 1]], dim=-1)
        dist = relpos.norm(dim=-1, keepdim=True)
        type_pair = nt.unsqueeze(2) * 3 + nt.unsqueeze(1)         # [B,N,N]
        z = self.interact(h_self, relpos, relvel, dist, type_pair)  # [B,N,D]

        # --- non-autoregressive decode: all 12 steps of local MoG params ---
        out = self.dec(torch.cat([z, team_emb], dim=-1))           # [B,N,HOR*K*5]
        out = out.reshape(B, N, HOR, K_MIX, 5)

        if self.training:
            # return local params as [HOR, B*N, K*5] to mirror v46 preds ordering (b*N + n)
            return out.permute(2, 0, 1, 3, 4).reshape(HOR, B * N, K_MIX * 5)

        # eval: weighted MoG mean in local -> global -> [HOR, B*N, 2]
        logit = out[..., 4]
        logw = logit - logit.logsumexp(dim=-1, keepdim=True)
        w = logw.exp().unsqueeze(-1)                               # [B,N,HOR,K,1]
        mu_local = (w * out[..., :2]).sum(dim=3)                   # [B,N,HOR,2]
        glob = pt_to_global(mu_local, c[:, :, None], s[:, :, None], o[:, :, None, :])
        return glob.permute(2, 0, 1, 3).reshape(HOR, B * N, 2)


class MoGLocalNLLLoss:
    """Timestep-weighted MoG NLL computed in each agent's LOCAL frame."""
    def compute(self, pred, target, frames):
        c, s, o = frames
        B, Tt, N, _ = target.shape
        tgt_g = target[:, :, :, :2]                                # [B,HOR,N,2]
        tgt_l = pt_to_local(tgt_g, c[:, None, :], s[:, None, :], o[:, None, :, :])
        tgt = tgt_l.permute(1, 0, 2, 3).reshape(Tt, B * N, 2)      # [HOR, B*N, 2]

        pred = pred.reshape(Tt, B * N, K_MIX, 5)
        mu = pred[:, :, :, :2]
        log_sigma = pred[:, :, :, 2:4].clamp(-3, 3)
        logit_w = pred[:, :, :, 4]
        log_w = logit_w - logit_w.logsumexp(dim=-1, keepdim=True)
        var = (log_sigma * 2).exp()
        tgt_k = tgt.unsqueeze(2).expand_as(mu)
        diff = tgt_k - mu
        log_prob_k = -0.5 * (diff ** 2 / var + log_sigma * 2 + math.log(2 * math.pi))
        log_prob_k = log_prob_k.sum(dim=-1)
        log_mix = torch.logsumexp(log_w + log_prob_k, dim=-1)      # [HOR, B*N]
        weights = torch.linspace(1.0, 2.0, Tt, device=pred.device)
        total = sum(weights[t] * (-log_mix[t].mean()) for t in range(Tt))
        return total / weights.sum()


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

        model = NBAv65Model().to(self.device)
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

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v65.pth')
        self.model = model
