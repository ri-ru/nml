
## [2026-05-01 now] nba_v3.py
**Hypothesis:** Added explicit velocity features (vx, vy) as inputs (6 features instead of 4), giving the model direct access to motion information as an inductive bias.
**Score:** 6.0294

## [2026-05-01] nba_v4.py
**Hypothesis:** Physics-residual predictor: constant-velocity baseline + learned residual correction, non-autoregressive decoding from final hidden state.
**Score:** 6.6208

## [2026-05-01] nba_v5.py
**Hypothesis:** Larger model (state_dim=192, 3 layers) + scheduled teacher forcing (0.8→0.0) to eliminate training error accumulation. Expected teacher forcing would close train/inference gap.
**Score:** 11.9522
**Lesson:** Teacher forcing HURT — model learned to predict given true positions but failed autoregressively at inference. Distribution shift between train (TF) and inference (no TF) was too large even with scheduled decay. Removed from v6.

## [2026-05-01] nba_v6.py
**Hypothesis:** Add ball-relative position/velocity features to each node ([rel_x, rel_y, rel_vx, rel_vy] = distance from ball) and type-aware attention bias for (ball↔player, teammate, opponent) pairs, giving the model explicit social graph structure.
**Score:** 4.3609 ← NEW BEST (improved from 6.0294)

## [2026-05-01] nba_v7.py
**Hypothesis:** Replace GRU context encoder with Temporal Transformer (self-attention across 8 timesteps per player + spatial attention between players). Longer-range temporal dependencies over context.
**Score:** 4.6269 (worse than v6's 4.3609 — Transformer overhead doesn't pay off for short 8-frame context; GRU handles it well)

## [2026-05-01] nba_v8.py
**Hypothesis:** Predict displacement Δpos instead of absolute position — constant-position default (Δpos=0) is better inductive bias than predicting raw coordinates.
**Score:** 5.3850 (worse than v6's 4.36 — accumulated error in Δpos is harder to train than absolute position, constant-position is too conservative for moving athletes)

## [2026-05-01] nba_v9.py
**Hypothesis:** Autoregressive constant-velocity baseline + residual (pos=prev+vel+residual) with scaled model (state_dim=192, 3 layers). Physics prior stronger than constant-position.
**Score:** 5.0170 (worse than v6's 4.36 — CV baseline still less accurate than direct absolute position prediction; scaling up model without new features doesn't help)

## [2026-05-01] nba_v10.py
**Hypothesis:** Add formation-aware features: each player's position relative to own team centroid, opponent team centroid, and ball-direction alignment (cosine of velocity vs direction-to-ball). These encode global tactical structure beyond dyadic ball-relative features.
**Score:** 4.3651 (no improvement over v6's 4.3609 — centroid features redundant with what typed attention already captures; ball alignment cosine adds noise without signal)

## [2026-05-01] nba_v11.py
**Hypothesis:** Lower LR (5e-4 vs v6's 1e-3) + more epochs (40 vs 35) for more stable convergence. CosineAnnealingLR over 40 epochs. Exact v6 architecture otherwise.
**Score:** 4.7674 (WORSE than v6's 4.3609 — lower LR actually hurt; CosineAnnealing over 40 epochs decays too slowly, model underfits. v6's lr=1e-3 is already the right rate.)

## [2026-05-01] nba_v12.py
**Hypothesis:** Add K=1 nearest-teammate and K=1 nearest-opponent relative position/velocity features (8 new features per player). Each player explicitly knows who guards them (nearest opp) and nearest pass target (nearest teammate). Ball nodes zeroed out.
**Score:** 4.1568 ← NEW BEST (beat v6's 4.3609 by ~4.7%) — dyadic social features help; nearest opponent context especially useful for predicting defensive/offensive movement.

## [2026-05-01] nba_v13.py
**Hypothesis:** Combine ALL proven feature sets: v6 ball-relative + v12 nearest-social + v10 team centroids + ball-alignment cosine. Use slightly larger model (state_dim=160) to handle 21 features.
**Score:** 4.6233 (worse than v12's 4.1568 — centroid and ball-align features add noise on top of v12; more features dilutes the useful social signal. Keep just ball-rel + nearest-social.)

## [2026-05-01] nba_v14.py
**Hypothesis:** Directional attention: use full 2D relative position vector (dx, dy, dist) as attention bias instead of scalar distance. Lets model distinguish left/right directionality in spatial relationships.
**Score:** NaN — gradient explosion after epoch 3. When two predicted positions converge (dist→0), the direction normalization diff/(dist+1e-6) creates near-zero denominators with large gradients. Fix: use F.normalize + log1p(dist).

## [2026-05-01] nba_v15.py
**Hypothesis:** Extend v12's K=1 nearest-social to K=2 (2 nearest teammates + 2 nearest opponents). Second nearest players often relevant: help defender, alternate pass target.
**Score:** 4.0356 ← NEW BEST (beat v12's 4.1568 by ~3.4%) — K=2 social features improve over K=1; trend suggests more nearest-neighbor context helps.

## [2026-05-01] nba_v16.py
**Hypothesis:** Fix v14's NaN by using F.normalize (stable unit vector) + log1p(dist) for directional attention bias. Thought F.normalize avoids division by near-zero.
**Score:** NaN — epoch 1. Root cause: F.normalize gradient ∂normalize/∂x = (I - vv^T)/max(||x||, eps). When ||x|| < eps=1e-12, gradient ≈ I/1e-12 = 1e12, causing explosion. F.normalize is NOT safe for backprop when inputs can be near-zero. Directional attention fundamentally incompatible with autoregressive decoding where predicted positions can coincide.

## [2026-05-01] nba_v17.py
**Hypothesis:** Extend v15's K=2 nearest-social to K=3 (3 nearest teammates + 3 nearest opponents). 3rd nearest opponent = help-help defender in zone defense.
**Score:** 4.0879 (WORSE than v15's 4.0356 — K-nearest trend does NOT continue past K=2; 3rd neighbor adds noise. K=2 is the sweet spot for hard social features.)

## [2026-05-01] nba_v18.py
**Hypothesis:** Context trajectory summary: prepend mean_vel + last_vel from 8-frame context as persistent decoder features, giving the model each player's "motion character" explicitly.
**Score:** 4.0847 (worse than v15's 4.0356 — explicit mean_vel/last_vel context features add noise; GRU hidden state already captures motion character implicitly)

## [2026-05-01] nba_v19.py
**Hypothesis:** V15 architecture (K=2 social + ball-rel) trained for 100 epochs with CosineAnnealingLR(T_max=100). Longer optimization path at same lr=1e-3.
**Score:** 3.8371 local / **4.22201 Kaggle** ← NEW BEST on both metrics (~5.4% improvement over v15 locally, ~3.1% on Kaggle).
**Lesson:** Training duration is the most impactful lever found so far — far larger gain than any feature engineering since v15. v11 showed that lower LR + 40 epochs HURT (lr=5e-4 + CosineAnnealing too slow). v19 shows that keeping lr=1e-3 but extending T_max to 100 lets the model find a much better minimum. The consistent local→Kaggle gap (~0.38) is larger than for simpler models, suggesting slight overfitting — but ranking is preserved. Next priority: retrain best feature configs (v20, v21) with 100 epochs.

## [2026-05-02] nba_v20.py
**Hypothesis:** K=1 hard-nearest + soft distance-weighted global attention over ALL teammates/opponents. Same 16 social features as v15 but soft component captures full formation context vs v15's 2nd nearest only. Learned temperature parameter. No NaN risk (uses distances not direction vectors).
**Score:** 4.0292 local (35 epochs) — marginally edges v15's 4.0356 by 0.16%. Not tested at 100 epochs yet.
**Lesson:** Soft global attention helps slightly over hard K=2 but the gain is tiny compared to epoch count. Retrain v20 with 100 epochs to see if it beats v19.

## [2026-05-02] nba_v21.py
**Hypothesis:** GNN edge-MLP message passing replaces typed distance-biased attention in hidden state. Edge function: MLP(h_i, h_j, rel_pos, rel_vel, type_emb) → message. Distance-weighted aggregation. Raw rel_pos as MLP input avoids NaN (no normalization).
**Score:** 4.1226 local (35 epochs) — WORSE than v15 (4.0356) and v20 (4.0292). GNN message passing does not improve over typed distance-biased attention at this epoch count; edge MLPs may need more training to converge. Retrain with 100 epochs before concluding it's a dead end.

## [2026-05-02] nba_v24.py
**Hypothesis:** V20 architecture (K=1 hard-nearest + soft distance-weighted global attention over all teammates/opponents, learned temperature) trained for 100 epochs with CosineAnnealingLR(T_max=100), lr=1e-3. Apples-to-apples comparison with v19 (v15 arch × 100 epochs) — testing whether soft global attention outperforms hard K=2 at same epoch budget.
**Score:** 3.8269 local ← **NEW BEST** (beat v19's 3.8371 by 0.3%). Val loss converged to ~0.0164 vs v23/v15-equivalent's ~0.0176 at same epoch.
**Lesson:** Soft global attention (v20 arch) does beat hard K=2 (v15 arch) at 100 epochs. The improvement is small (+0.3%) but consistent — every entity gets a soft formation-aware context in addition to the nearest hard neighbor. Next: train v20 arch at 150 epochs (v25) to continue the epoch-scaling trend.

## [2026-05-02] nba_v23.py
**Hypothesis:** V15 architecture (K=2 social + ball-rel) trained for 150 epochs with CosineAnnealingLR(T_max=150), lr=1e-3. Testing whether the epoch-scaling trend continues beyond 100 epochs.
**Score:** 3.7724 local ← **NEW BEST** (beat v19's 3.8371 by 1.7%, beat v24's 3.8269 by 1.4%).
**Lesson:** Epoch scaling continues past 100 epochs — 150ep gives another meaningful gain on top of 100ep. The CosineAnnealingLR curve at T_max=150 gives the optimizer more time in the low-LR refinement phase. Next logical step: v20 arch × 150 epochs (v25) to see if soft attention + 150ep beats v23.

## [2026-05-02] ensemble_v19_v23_v24
**Hypothesis:** Average predicted positions from v19 (3.8371 local), v23 (3.7724 local), v24 (3.8269 local) — three diverse models (different archs, different epoch counts) whose errors should partially cancel.
**Action:** Equal-weight average of solution_v19.csv + solution_v23.csv + solution_v24.csv → solution_ensemble_v19_v23_v24.csv. 1243 test sequences × 264 prediction columns.
**Score:** Pending Kaggle submission. Expected ~4.1–4.15 (vs best single-model Kaggle 4.22201).

## [2026-05-02] nba_v26.py
**Hypothesis:** V24 architecture (K=1 hard-nearest + soft global social attention) + court symmetry augmentation (x-flip p=0.5, y-flip p=0.5 independently). Court is centered at origin (mu≈0), so negating normalized coords is exact reflection. Augmentation doubles/quadruples effective training data, should close train→Kaggle gap. 100 epochs.
**Score:** 3.7932 local ← beats v24 (3.8269, same arch, no aug) by 0.9% at same epoch count. Worse than v23 (3.7724, 150ep, no aug). Local score is train==val so augmentation making training harder explains the gap vs v23.
**Lesson:** Augmentation helps at same epoch count. Next logical step: v26 arch × 150 epochs to see if aug + more training beats v23. The real test is Kaggle — augmentation should reduce the ~0.38 local→Kaggle gap even if local score stays similar.

## [2026-05-02] nba_v22.py — 35 epochs (exploration)
**Hypothesis:** Decoder self-attention over all previous decode steps (cross-attend h_d1..h_{t-1} at each step t). Gives decoder explicit memory of full decoded trajectory. v20 features + step-distance positional bias.
**Score:** 4.3494 ← WORSE than v15 baseline (4.0356). Dead end.
**Lesson:** Decoder self-attention adds overhead without benefit. The GRU hidden state already captures trajectory history implicitly. Cross-attending to previous decode states is redundant and adds noise. Do NOT pursue further.

## [2026-05-02] nba_v27.py — 35 epochs (exploration)
**Hypothesis:** Larger capacity: state_dim=256, 3 GRU+attn layers (vs v24's 128, 2 layers). Tests Tier-D.
**Score:** 6.9413 — far worse than v24 (3.8269). Dead end at 35 epochs.
**Lesson:** Large model is a "late bloomer" — loss stuck at ~0.55 for 17 epochs before phase transition at epoch 18 (sudden collapse to 0.10 then 0.07). Only had 17 actual learning epochs, massively underfitted. Local→score ratio is 282 vs the usual 225, confirming poor generalization at this stage. To be useful, would need 100-150 epochs. Not worth pursuing given v26 (augmentation) is the higher-priority direction. If capacity scaling is tried again, use LR warmup (lr_start=1e-4 → 1e-3 over 20 epochs) to avoid the dead zone.

## [2026-05-02] nba_v28.py — 35 epochs (exploration)
**Hypothesis:** V24 arch + acceleration features (accel = vel_t - vel_{t-1}, 2 extra features). Input dim 32→34. Tests Tier-H.
**Score:** 4.0908 — slightly worse than v20 baseline (4.0292) at same 35 epochs. Dead end.
**Lesson:** Explicit acceleration doesn't help — velocity already encodes first-order motion, and GRU can implicitly differentiate velocity over time. Adding accel slightly dilutes the useful features. Do not pursue.

## [2026-05-02] nba_v29.py — 35 epochs (exploration)
**Hypothesis:** V24 arch with context_size=16 (0.8s of history vs default 0.4s). Tests Tier-K.
**Score:** 5.5308 — much worse than v20 baseline (4.0292) at same epochs. Dead end.
**Lesson:** Longer context hurts. Two compounding reasons: (1) filtering sequences shorter than window_size=28 removes training data, reducing coverage; (2) the GRU processes 16 frames before decoding — more context steps may push the hidden state into a regime harder to decode from. Context=8 is the right window for this architecture and dataset. Do not pursue longer context.

## [2026-05-02] nba_v30.py — 150 epochs (long run)
**Hypothesis:** V26 architecture (v24 arch + court symmetry augmentation) trained for 150 epochs. V26 at 100ep scored 3.7932, beating v24 (3.8269) by 0.9%. V23 at 150ep scored 3.7724. Hypothesis: aug + 150ep beats both.
**Score:** 3.7535 local ← **NEW BEST** (beat v23's 3.7724 by 0.5%, beat v26-100ep's 3.7932 by 1.0%). Saved as model_v26.pth (overwrites 100-ep checkpoint).
**Lesson:** Augmentation + 150ep is the current winning combination. Confirms both axes: (1) epoch scaling continues past 100ep, (2) augmentation adds value on top of raw epoch count. Next: submit TTA ensemble (v19+v23+v24+v30) to Kaggle; start v31 = v26 arch × 200 epochs or try dropout regularization.

## [2026-05-02] nba_v31.py — 200 epochs
**Hypothesis:** V26 architecture (v24 + court symmetry augmentation) × 200 epochs. Epoch scaling trend: 100ep→3.7932, 150ep→3.7535. Testing if improvement continues to 200ep.
**Score:** 3.7319 local ← **NEW BEST** (beat v30's 3.7535 by 0.6%). Saved as model_v31.pth.
**Lesson:** Epoch scaling continues to 200 epochs: gains are decelerating (100→150: -0.0397, 150→200: -0.0216) but still meaningful. Worth trying 250 epochs. Also add v31 to TTA ensemble.

## [2026-05-02] nba_v32.py — 150 epochs
**Hypothesis:** V26 arch + dropout(p=0.15) on GRU hidden states and projection head × 150 epochs. All prior models have zero dropout; only weight_decay=5e-4. Hypothesis: dropout closes local→Kaggle generalization gap by reducing overfitting.
**Score:** 4.8190 — MUCH WORSE than v31 (3.7319). Clear dead end.
**Lesson:** Dropout is fundamentally incompatible with autoregressive GRU decoding. At inference, the GRU hidden state is used as the sole memory across 12 decode steps. Dropout on this state (even p=0.15) corrupts the accumulated representation and causes error to compound over steps. val_loss stayed at ~0.020 vs ~0.016 without dropout — the model never converged to the same quality. Do NOT use dropout on GRU hidden states in autoregressive models. Weight_decay (L2) is the right regularizer here.

## [2026-05-02] nba_v33.py — 250 epochs
**Hypothesis:** V26 arch × 250 epochs. Epoch scaling: 100→3.7932, 150→3.7535, 200→3.7319.
**Score:** 3.7471 — slightly WORSE than 200ep (3.7319). Epoch scaling plateau for MSE models at ~200ep. This was the last MSE experiment; NLL breakthrough made all MSE scaling obsolete.

## [2026-05-02] nba_v34.py — superseded
**Hypothesis:** V26 arch × 200 epochs + EMA weights. Never run — NLL breakthrough (v38) made all MSE models obsolete before this could run.
**Score:** N/A — skipped.

---

## [2026-05-05] nba_v37.py — 35 epochs — NLL ERA BEGINS
**Hypothesis:** Non-autoregressive Transformer + Gaussian NLL loss. First experiment with probabilistic output head: predict (mu_x, mu_y, log_sigma_x, log_sigma_y) per player per step, optimize NLL instead of MSE. Non-autoregressive: decode all 12 steps from final encoder hidden state.
**Score:** 3.8641 — beats entire MSE era at 35 epochs. NLL hypothesis confirmed immediately.
**Lesson:** Probabilistic loss is the breakthrough. Even a worse architecture (non-autoregressive Transformer) scores better than our best 200-epoch MSE model just by switching the loss.

## [2026-05-05] nba_v38.py — 35 epochs
**Hypothesis:** Autoregressive GRU (our best MSE architecture) + Gaussian NLL. Expected to beat non-autoregressive v37.
**Score:** 3.4355 local / **3.68213 Kaggle** — NEW BEST. Beats v37 (3.8641) and beats all MSE models. With 4-variant TTA on Kaggle: 3.68213. This is the base for all future experiments.
**Lesson:** NLL + GRU autoregressive = best combination. Non-autoregressive doesn't work well for this task (already knew this from v4/v9, now confirmed for Transformer too). The local→Kaggle gap (~7%) appears stable and is likely the true signal, not overfitting.

## [2026-05-06] nba_v39.py — 35ep → 150ep
**Hypothesis:** Bigger GRU (state_dim=192, 3 layers) + NLL. More capacity should extract more from NLL.
**Score:** 6.37 at 35ep (late bloomer), 3.8514 at 150ep. WORSE than v38 (3.4355) even at 4× epoch budget.
**Lesson:** Model capacity is not the bottleneck. The 128d 2-layer GRU is enough for this problem. Bigger models need far more training to converge and still don't beat the smaller one. Dead end for capacity scaling.

## [2026-05-06] nba_v40.py — 35ep → 100ep
**Hypothesis:** Transformer encoder + NLL × 100 epochs. V37 used non-autoregressive Transformer; this is autoregressive Transformer decoder.
**Score:** 3.9194 at 35ep, 3.4225 at 100ep. Marginally better than v38 at 100ep, but v38 wasn't trained to 100ep yet — this doesn't isolate the Transformer benefit.
**Lesson:** Transformer + NLL at 100ep (3.4225) is comparable to GRU + NLL at 35ep (3.4355). The GRU architecture is more sample-efficient. Not worth pursuing Transformer further given GRU trains faster and generalizes better at short budgets.

## [2026-05-06] nba_v41.py — 35 epochs
**Hypothesis:** V38 arch + timestep-weighted NLL (linear weights 1.0→2.0 across 12 steps, emphasizing far-future predictions). Closer timesteps are easier; far-future ones matter more for the evaluation metric.
**Score:** 3.4323 — essentially same as v38 (3.4355). Timestep weighting doesn't help at 35 epochs.
**Lesson:** The model doesn't have enough training capacity at 35ep to act on the extra gradient signal for far-future steps. Worth retesting at 100ep+.

## [2026-05-06] nba_v42.py — 35ep → 100ep → 250ep
**Hypothesis:** V38 arch + K=2 nearest neighbor social features (2 nearest teammates + 2 nearest opponents, rel_pos + rel_vel). 24 social features total. Same feature set that helped in the MSE era but never tested with NLL.
**Score:** 3.4281 at 35ep (barely helps), **3.2099 at 100ep (NEW BEST by large margin)**, **3.0558 at 250ep (NEW BEST)**.
**Lesson:** K=2 social features need training time to activate. At 35ep they add almost nothing; at 100ep they unlock a large gain. Epoch scaling compounds with social features — these two axes multiply rather than add. At 250ep, still improving: 100ep→3.2099, 250ep→3.0558 (-4.6%).

## [2026-05-06] nba_v43.py — 35ep → 100ep → 250ep
**Hypothesis:** V42 arch + timestep-weighted NLL (1.0→2.0 weights). Testing whether v41's null result was epoch-limited.
**Score:** 3.4512 at 35ep (worse than v42's 3.4281 — weighting hurts at short budget), **3.1936 at 100ep (NEW BEST)**, **3.0395 at 250ep (NEW BEST)**.
**Lesson:** Timestep weighting is epoch-dependent: hurts at 35ep, helps at 100ep and 250ep. The model needs enough capacity (epochs) to learn to allocate attention to far-future timesteps. At 250ep, beats v42×250ep (3.0558) by 0.5%.

## [2026-05-06] nba_v44.py — 100ep → 250ep — MoG BREAKTHROUGH
**Hypothesis:** V43 arch + Mixture of Gaussians K=2 output instead of single Gaussian. Each player at each timestep: 2 Gaussian components with learned mixing weights. Loss: MoGNLL via logsumexp. Inference: weighted mean of component means. Motivation: basketball is bimodal — player driving toward defender can go left OR right. Single Gaussian predicts the average (wrong); MoG can represent both modes.
**Score:** 3.1177 at 100ep (beats v43×100ep=3.1936 by 3.9%), **2.8963 at 250ep — MASSIVE NEW BEST (beats v43×250ep=3.0395 by 4.8%)**.
**Lesson:** MoG is the largest single architectural gain since switching to NLL. The bimodal hypothesis is confirmed. K=2 components capture binary decisions (left/right, drive/pass). Compounds strongly with epoch scaling. Current best model.

## [2026-05-06] nba_v45.py — 300 epochs
**Hypothesis:** V42 arch (K=2 + Gaussian NLL, no timestep weighting, no MoG) × 300 epochs with T_max=300. Control experiment: does a proper 300-epoch LR schedule beat v43×250ep (which used T_max=250 effectively)? Tests the pure epoch-scaling ceiling for the base arch.
**Score:** 3.0177 — beats v43×250ep (3.0395) by 0.6%. T_max=300 does help slightly vs running past T_max. But MoG (v44×250ep=2.8963) is far better.
**Lesson:** Proper LR schedule (T_max=epochs) matters at the margin. But the architectural improvement from MoG dwarfs the scheduling gain. Future long runs should always set T_max=epochs.

---

## Next Steps — Priority List (2026-05-06, updated)

Current best: **v44×250ep local 2.8963** / Kaggle TBD (expected ~3.10 based on 7% gap). Best Kaggle so far: ~3.44785 (v42×100ep TTA).

### Tier 1 — Run immediately

**A. v44 × 300ep** (v46)
Same MoG K=2 arch with T_max=300. Every model we've run longer has improved. v44×250ep=2.8963 — estimate ~2.85 at 300ep. v45 showed T_max=300 helps vs running past T_max=250. Highest expected gain of anything in the queue.

**B. Ensemble TTA: v43×250ep + v44×250ep**
Average the two TTA CSV outputs. Zero training cost. Two models with different uncertainty representations (Gaussian vs MoG) — their residual errors should be less correlated than two runs of the same arch. Expected gain: 0.02–0.05 local.

### Tier 2 — Architectural exploration

**C. MoG K=3**
v44 uses K=2 components (captures binary decisions). K=3 might capture three-way choices (drive left, drive right, kick out to shooter). Same arch, change K_MIX=3 in nba_v44.py, test at 100ep first.

**D. MoG K=2 without timestep weighting**
v44 = v43 + MoG, where v43 already has timestep weighting. Does MoG benefit from timestep weighting, or does the mixture already solve the far-future problem? Test: v42 + MoG (no weighting) at 100ep, compare to v44×100ep (3.1177).

### Tier 3 — If Tier 1-2 exhausted

**E. Larger ensemble**: TTA of v42×250ep + v43×250ep + v44×250ep all averaged. More model diversity = more variance reduction.

**F. Decoder uncertainty propagation**: Currently MoG inference uses weighted mean of component means, ignoring the mixture uncertainty in autoregressive steps. Try feeding the full MoG sample (or top-component mean) as the next-step input instead of the weighted mean — could give the model a better signal about which mode it's "in."

### Submission strategy (updated)

1. Submit v44×250ep TTA immediately (current best)
2. Submit ensemble TTA (v43+v44) as soon as script is ready
3. Submit solo v44×250ep to isolate TTA effect
4. When v46 (v44×300ep) finishes, TTA and submit
