# Model Results

| Time | Model | Local Score | Kaggle Score | Notes |
|------|-------|-------------|--------------|-------|
| 2026-05-01 10:55 | GRU baseline | 80.1490 | — | 10 epochs, state_dim=32 |
| 2026-05-01 10:55 | GAT | 24.7375 | — | 50 epochs, state_dim=64, 4 heads |
| 2026-05-01 | nba_v3 | 6.0294 | — | Explicit velocity features (vx, vy) as inputs |
| 2026-05-01 | nba_v4 | 6.6208 | — | Physics-residual, non-autoregressive decoder |
| 2026-05-01 | nba_v5 | 11.9522 | — | Teacher forcing — HURT badly |
| 2026-05-01 | nba_v6 | 4.3609 | 4.63615 | Ball-relative features + typed attention — big jump |
| 2026-05-01 | nba_v7 | 4.6269 | — | Temporal Transformer context encoder — worse |
| 2026-05-01 | nba_v8 | 5.3850 | — | Predict displacement Δpos — worse |
| 2026-05-01 | nba_v9 | 5.0170 | — | Constant-velocity residual + larger model — worse |
| 2026-05-01 | nba_v10 | 4.3651 | — | Team centroid features — no improvement |
| 2026-05-01 | nba_v11 | 4.7674 | — | Lower LR (5e-4) + 40 epochs — worse |
| 2026-05-01 | nba_v12 | 4.1568 | 4.46251 | K=1 nearest teammate+opponent social features |
| 2026-05-01 | nba_v13 | 4.6233 | — | All features combined (centroid + social) — noise |
| 2026-05-01 | nba_v14 | NaN | — | Directional attention — gradient explosion at dist→0 |
| 2026-05-01 | nba_v15 | 4.0356 | 4.35509 | K=2 nearest social features |
| 2026-05-01 | nba_v16 | NaN | — | F.normalize directional fix — still NaN (∂normalize/∂x explodes at dist→0) |
| 2026-05-02 09:52 | nba_v20 | 4.0292 | — | K=1 hard + soft global social attention, learned temperature (35 epochs) |
| 2026-05-02 09:55 | nba_v18 | 4.0847 | — | Context mean_vel+last_vel as decoder input — noise (35 epochs) |
| 2026-05-02 09:58 | nba_v17 | 4.0879 | — | K=3 nearest social — K trend saturated, worse than K=2 (35 epochs) |
| 2026-05-02 10:25 | nba_v21 | 4.1226 | — | GNN edge-MLP message passing — worse than v15 (35 epochs) |
| 2026-05-02 10:52 | **nba_v19** | **3.8371** | **4.22201** | **v15 arch × 100 epochs — NEW BEST (local + Kaggle)** |
