# Model Results

| Time | Model | Local Score | Kaggle Score | Notes |
|------|-------|-------------|--------------|-------|
| 2026-05-01 11:49 | Baseline GRU | 54.0075 | — | Naive GRU, no interaction between players, no velocity features |
| 2026-05-01 11:53 | GAT | 23.2694 | — | Graph Attention Network: players interact via edges; no temporal model beyond GRU hidden state |
| 2026-05-01 12:11 | v2 dist-GAT | 9.2611 | — | Distance-biased multi-head attention between players (replaces plain GAT); state_dim=128, 2 layers |
| 2026-05-01 13:33 | nba_v3 | 6.0294 | — | +Explicit velocity inputs (vx,vy) — 6 features instead of 4; autoregressive decoding; same dist-biased attention |
| 2026-05-01 13:42 | nba_v4 | 6.6208 | — | Non-autoregressive physics residual: constant-velocity baseline + GRU correction decoded all at once — worse than v3 |
| 2026-05-01 14:08 | nba_v5 | 11.9522 | — | Scheduled teacher forcing (0.8→0) + larger model (state_dim=192, 3 layers) — teacher forcing created train/inference gap, badly hurt |
| 2026-05-01 16:59 | nba_v6 | 4.3609 | 4.63615 | +Ball-relative features for each entity (rel_x,rel_y,rel_vx,rel_vy to ball) + type-pair attention bias (ball/teammate/opponent typed edges) — new best |
| 2026-05-01 17:10 | nba_v7 | 4.6269 | — | Replaced GRU context encoder with Temporal Transformer (self-attn across 8 frames per player) + GRU decoder — no gain over v6, Transformer overhead not worth it for 8-frame context |
| 2026-05-01 17:25 | nba_v8 | 5.3850 | — | Displacement prediction (pos=prev+Δpos) on v6 arch — constant-position default too conservative, accumulated Δpos error worse than absolute position prediction |
| 2026-05-01 17:45 | nba_v9 | 5.0170 | — | Autoregressive CV prior + residual (pos=prev+vel+Δ) + scaled model (state_dim=192, 3L) — CV baseline still worse than direct absolute prediction; larger model without new features doesn't help |
| 2026-05-01 18:09 | nba_v10 | 4.3651 | — | +Team centroid offsets (rel to own/opp centroid) + ball-direction alignment cosine — no improvement over v6; global formation features redundant given typed attention already captures team structure |
| 2026-05-01 18:18 | nba_v11 | 4.7674 | — | V6 arch with lr=5e-4 (half of v6) + 40 epochs — lower LR hurt: CosineAnnealing decays too slowly, model underfits; v6's lr=1e-3 is already well-tuned |
| 2026-05-01 18:29 | nba_v12 | 4.1568 | 4.46251 | +Nearest-teammate and nearest-opponent relative pos/vel features (8 new features on v6 base) — dyadic social context helps; NEW BEST |
| 2026-05-01 19:03 | nba_v13 | 4.6233 | — | All features combined (v6+v12 social+v10 centroids+ball-align) in state_dim=160 — worse than v12; centroid+ball-align features add noise, more features ≠ better with same capacity |
| 2026-05-01 19:02 | nba_v14 | nan | — | Directional attention (dx,dy,dist) vs scalar dist — NaN after epoch 3: gradient explosion from direction normalization when predicted positions converge (dist→0) |
| 2026-05-01 19:58 | nba_v15 | 4.0356 | 4.35509 | K=2 nearest teammate + K=2 nearest opponent relative pos/vel (16 social features) — more social context helps; NEW BEST |
| 2026-05-01 20:45 | nba_v16 | nan | — | Stable directional attention (F.normalize + log1p) on v12 base — NaN epoch 1: F.normalize gradient explodes at diff→0 (∂normalize/∂x = I/eps when x→0). Directional attention fundamentally broken for autoregressive decoding. |
| 2026-05-02 09:52 | nba_v20 | 4.0292 | — | K=1 hard-nearest + soft distance-weighted global attention over all teammates/opponents (16 social features, learned temperature) — marginal gain over v15's hard K=2; 35 epochs |
| 2026-05-02 09:55 | nba_v18 | 4.0847 | — | +Context trajectory summary (mean_vel + last_vel from 8 context frames) as persistent decoder input — worse than v15; GRU hidden state already captures motion character implicitly; 35 epochs |
| 2026-05-02 09:58 | nba_v17 | 4.0879 | — | K=3 nearest teammate + K=3 nearest opponent (24 social features) — K-nearest trend saturates at K=2; 3rd neighbor adds noise; 35 epochs |
| 2026-05-02 10:25 | nba_v21 | 4.1226 | — | GNN edge-MLP message passing replaces typed attention: MLP(h_i,h_j,rel_pos,rel_vel,type_emb)→message, distance-weighted aggregation — worse than v15 at 35 epochs; may need more training to converge |
| 2026-05-02 10:52 | **nba_v19** | **3.8371** | **4.22201** | **v15 arch (K=2 social + ball-rel) × 100 epochs, CosineAnnealingLR(T_max=100), lr=1e-3 — biggest improvement yet; longer schedule at same LR outperforms all feature engineering; NEW BEST** |
| 2026-05-02 13:10 | **nba_v24** | **3.8269** | — | **v20 arch (K=1 hard + soft global social attention, learned temperature) × 100 epochs, CosineAnnealingLR(T_max=100), lr=1e-3 — soft global attention beats v15 hard K=2 at same epoch budget; NEW BEST** |
| 2026-05-02 13:30 | **nba_v23** | **3.7724** | — | **v15 arch (K=2 social + ball-rel) × 150 epochs, CosineAnnealingLR(T_max=150), lr=1e-3 — 150ep beats 100ep (3.8371); epoch scaling still improving; NEW BEST** |
| 2026-05-02 16:55 | nba_v22 | 4.3494 | — | Decoder self-attention over previous decode steps (cross-attend h_d1..h_{t-1}) on v20 features — dead end; GRU hidden state already captures trajectory history implicitly |
| 2026-05-02 17:25 | **nba_v26** | **3.7932** | — | **v24 arch + court symmetry augmentation (x-flip p=0.5, y-flip p=0.5) × 100 epochs — beats v24 (3.8269) by 0.9% at same epoch count; augmentation helps** |
| 2026-05-02 17:25 | nba_v27 | 6.9413 | — | v24 arch scaled to state_dim=256, 3 layers × 35 epochs — dead end at 35ep; large model stuck at loss~0.55 for 17 epochs (late bloomer), only 17 useful learning epochs; needs 100+ epochs |
| 2026-05-02 17:45 | nba_v28 | 4.0908 | — | v24 arch + acceleration features (accel=vel_t-vel_{t-1}, 2 extra features) × 35 epochs — no gain; GRU implicitly differentiates velocity, explicit accel redundant |
| 2026-05-02 18:16 | nba_v29 | 5.5308 | — | v24 arch with context_size=16 (0.8s history vs 0.4s) × 35 epochs — hurts; filtering seqs shorter than 28 frames reduces training data; longer context degrades GRU hidden state quality for decoding |
| 2026-05-02 19:30 | **nba_v30** | **3.7535** | — | **v26 arch (v24+augmentation) × 150 epochs, CosineAnnealingLR(T_max=150) — augmentation + 150ep beats aug+100ep (3.7932) and v23 (3.7724); NEW BEST. Saved as model_v26.pth** |
| 2026-05-02 21:56 | **nba_v31** | **3.7319** | — | **v26 arch (aug + v24) × 200 epochs, CosineAnnealingLR(T_max=200) — epoch scaling continues: 100ep→3.7932, 150ep→3.7535, 200ep→3.7319; NEW BEST** |
| 2026-05-02 23:41 | nba_v32 | 4.8190 | — | v26 arch + dropout(p=0.15) × 150 epochs — DEAD END: dropout corrupts GRU hidden state for autoregressive rollout; val_loss stayed ~0.020 vs 0.016 without dropout; 29% regression |
| 2026-05-02 23:45 | nba_v33 | pending | — | v26 arch × 250 epochs — epoch scaling trend test (200ep→3.7319, testing if gains continue) |
| 2026-05-02 23:45 | nba_v34 | pending | — | v26 arch × 200 epochs + EMA (decay=0.999) — EMA shadow weights for inference; hypothesis: closes local→Kaggle gap |
| 2026-05-05 20:13 | nba_v33 | 3.7471 | agent loop (250 epochs) |
| 2026-05-05 20:24 | nba_v37 | 3.8641 | agent loop (35 epochs) |
| 2026-05-05 20:55 | nba_v38 | 3.4355 | **3.68213** | GRU + Gaussian NLL × 35ep — BEST MODEL; solo TTA beats all ensembles |
| 2026-05-05 21:36 | nba_v39 | 6.3680 | — | Bigger GRU (192d,3L) + NLL × 35ep — late bloomer, run 150ep |
| 2026-05-05 21:37 | nba_v40 | 3.9194 | — | Transformer + NLL × 35ep — run 100ep |
| 2026-05-06 09:31 | nba_v41 | 3.4323 | agent loop (35 epochs) |
| 2026-05-06 09:55 | nba_v42 | 3.4281 | agent loop (35 epochs) |
