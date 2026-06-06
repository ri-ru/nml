"""v72 — relational message-passing GNN. Replaces v46's TypeAwareDistanceBiasedAttention (biased
dot-product attention) with an explicit MESSAGE-PASSING GNN: messages are computed from node pairs +
edge geometry, conditioned on the relation type (teammate / opponent / player<->ball), aggregated,
and used to update each node. This is a strictly more expressive interaction than biased attention
(messages are learned functions of (h_i, h_j, edge), not just a softmax bias), and it's a textbook
relational GNN (R-GCN flavour, hand-rolled — no torch_geometric dependency).

Everything else is IDENTICAL to v46 (GRU temporal rollout, K=2 social input features, MoG K=2 head,
timestep-weighted NLL, 300ep cosine). Only the per-layer interaction module changes, so this is a
clean A/B: GNN message passing vs graph attention, same depth (2 layers = 2 propagation rounds).
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
from nba_v46 import (GRUCell, NBASampler, MoGNLLLoss, MultiStepMSE, NBAv46Model,
                     WEIGHTS_DIR, TEAM_EMB_DIM, K_MIX)


class RelationalMessagePassing(nn.Module):
    """One round of relational message passing over the 11-agent fully-connected graph.

    message  m_ij = MLP([h_i, h_j, edge_ij, rel_emb(type_i,type_j)])   (j -> i)
    aggregate a_i = mean_{j != i} m_ij
    update    h_i = LayerNorm(h_i + W a_i)        (residual, mirrors v46 attention's residual+norm)

    edge_ij = [rel_pos (j-i), rel_vel (j-i), dist]  (relative geometry; the GNN's geometric inductive
    bias). rel_emb gives each of the 9 ordered type pairs its own learned relation vector, so
    player<->ball / teammate / opponent messages differ in FUNCTION, not just a scalar bias.
    """
    def __init__(self, state_dim, num_types=3, rel_emb_dim=8, edge_dim=5, hidden=128):
        super().__init__()
        self.rel_emb = nn.Embedding(num_types * num_types, rel_emb_dim)
        self.msg = nn.Sequential(
            nn.Linear(2 * state_dim + edge_dim + rel_emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )
        self.upd = nn.Linear(state_dim, state_dim)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, h, pos, vel, node_type):
        B, N, D = h.shape
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)   # [B, Ni, Nj, 2] vector i->j
        rel_vel = vel.unsqueeze(1) - vel.unsqueeze(2)
        dist = rel_pos.norm(dim=-1, keepdim=True)
        edge = torch.cat([rel_pos, rel_vel, dist], dim=-1)            # [B,N,N,5]
        rel = node_type.unsqueeze(2) * 3 + node_type.unsqueeze(1)     # [B,N,N]
        rel_e = self.rel_emb(rel)                                     # [B,N,N,8]
        h_i = h.unsqueeze(2).expand(B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)
        m = self.msg(torch.cat([h_i, h_j, edge, rel_e], dim=-1))      # [B,N,N,D]
        eye = torch.eye(N, dtype=torch.bool, device=h.device)
        m = m.masked_fill(eye.view(1, N, N, 1), 0.0)                  # no self-message
        agg = m.sum(dim=2) / (N - 1)                                  # mean over neighbours [B,N,D]
        return self.norm(h + self.upd(agg))


class NBAv72Model(NBAv46Model):
    def __init__(self, state_dim=128, context_size=8, horizon_size=12, num_heads=4, num_layers=2):
        super().__init__(state_dim, context_size, horizon_size, num_heads, num_layers)
        # replace the attention interaction with relational message passing (same depth)
        self.attn_layers = None
        self.gnn_layers = nn.ModuleList([
            RelationalMessagePassing(state_dim) for _ in range(num_layers)
        ])

    def forward(self, X):
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
                h[i] = self.gnn_layers[i](
                    h[i].reshape(B, N, self.state_dim), pos, vel, node_type
                ).reshape(B * N, self.state_dim)
                layer_in = h[i]

            if self.context_size - 1 <= t < T - 1:
                out = self.proj(h[-1])
                out_k = out.reshape(B * N, K_MIX, 5)
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

        model = NBAv72Model().to(self.device)
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

        torch.save(model.state_dict(), WEIGHTS_DIR / 'model_v72.pth')
        self.model = model
