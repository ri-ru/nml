# NBA Trajectory Prediction: What We Tried

Competition: predict x,y positions for 11 players over the next 12 timesteps (0.6s) given 8 frames of context (0.4s). Scored by MSE in feet on Kaggle.

---

## Phase 1: Getting a working baseline (May 1 morning)

Started with the simplest possible thing: a GRU that processes each player independently, no interaction between players at all. Score was 54.0 - terrible but it runs.

Immediately added a Graph Attention Network (GAT) so players can interact through edges. Jumped to 23.3. Added distance-biased attention (closer players get higher attention weight) and got to 9.3. At this point the architecture is: encode each player's context frames with a GRU, let players attend to each other, then decode future positions autoregressively.

Added velocity as explicit input (vx, vy computed from position differences) and got to 6.0. This turned out to be important - the model needs to see velocity explicitly, not just positions.

## Phase 2: Feature engineering on the GRU-attention backbone (May 1 afternoon)

The architecture from here until May 5 is the same: GRU encoder + distance-biased multi-head attention + GRU decoder, all autoregressive.

Things that helped:
- **Ball-relative features** (position and velocity of each player relative to the ball): 4.36. This was a big jump. The ball is the most predictive thing - players move toward/away from the ball constantly.
- **Nearest neighbor social features** (relative position and velocity to the nearest teammate and nearest opponent): 4.16. Even just knowing who's closest to you and how fast they're moving helps.
- **K=2 nearest neighbors** (2 nearest teammates + 2 nearest opponents, 16 features total): 4.04. More social context helps.
- **Soft distance-weighted global attention** (instead of hard K-nearest, attend softly to all teammates weighted by distance): marginal improvement.
- **Court symmetry augmentation** (randomly flip x and/or y during training): consistent ~1-2% improvement. Basketball is symmetric so this is free data.
- **More training epochs** (100ep, 150ep, 200ep): huge. Going from 35ep to 200ep dropped MSE from ~4.0 to 3.73. Epoch scaling was more valuable than any individual feature.

Things that didn't help or hurt:
- Teacher forcing: caused train/inference gap, destroyed performance.
- Bigger model (state_dim=192, 3 layers) at 35ep: got stuck at 6.9 ("late bloomer" - needs 100+ epochs to converge).
- Displacement prediction (predict delta instead of absolute position): worse, error accumulates.
- Dropout: destroyed the GRU hidden state for autoregressive decoding, 29% regression.
- Directional attention (using dx, dy vectors instead of scalar distance): gradient explosion when players are close (dist->0). Tried two fixes, both failed with NaN. Dead end.
- Centroid features, acceleration features, longer context window: all hurt or no gain.

**Best at end of Phase 2: 3.73 (v31: v24 arch + augmentation, 200 epochs)**

## Phase 3: Gaussian NLL loss - the breakthrough (May 5)

Up to this point we were using MSE loss. The key insight: MSE pushes the model to predict the mean of the distribution, but the model should also learn *uncertainty*. When a player could plausibly go left or right, MSE loss will predict the middle (which is wrong in both cases). A probabilistic model can hedge appropriately.

Changed the output head to predict 4 values per player per timestep: (mu_x, mu_y, log_sigma_x, log_sigma_y). Trained with Gaussian NLL loss. At inference time only (mu_x, mu_y) are used for the MSE evaluation.

This dropped local MSE from 3.73 to **3.44 in just 35 epochs** (v38). On Kaggle it scored **3.68** with 4-variant TTA (original + x-flip + y-flip + xy-flip averaged).

Tried a non-autoregressive Transformer variant with NLL (v37): 3.86. Worse than the GRU. The Transformer encoder doesn't capture temporal dynamics as well as GRU hidden states for this problem.

Tried timestep-weighted NLL (linear weights 1.0 -> 2.0 across the 12 future steps, to emphasize far-future accuracy): 3.43, essentially the same as uniform NLL.

## Phase 4: Scaling what works (May 6 morning)

With NLL now the baseline, went back to testing features that we found useful before:

- **K=2 nearest neighbors + NLL** (v42, 35ep): 3.43. At 35 epochs, marginal improvement over plain NLL.
- **Transformer + NLL at 100 epochs** (v40): 3.42. Training longer helps.
- **K=2 nearest neighbors + NLL at 100 epochs** (v42x100): **3.21. New best by a huge margin.** The K=2 social features compound well with more training - at 35ep they barely helped, but at 100ep they unlock a lot.
- **Bigger GRU (192d, 3 layers) + NLL at 150 epochs** (v39): 3.85. Even with 150 epochs the big model underperforms the small one. More capacity doesn't help here.
- **K=2 nearest + timestep-weighted NLL at 35 epochs** (v43): 3.45. Timestep weighting hurts at short training budgets.
- **K=2 nearest + timestep-weighted NLL at 100 epochs** (v43x100): **3.19. New best.** Timestep weighting flips from hurting to helping once training is long enough. The later timesteps (further-future predictions) need more learning signal, and the model has enough epochs to act on it.
- **K=2 + NLL at 250 epochs** (v42x250): 3.06. Epoch scaling continues well past 100ep.
- **K=2 + timestep-weighted NLL at 250 epochs** (v43x250): **3.04. New best.** Timestep weighting keeps compounding at 250ep.
- **v42 arch retrained 300 epochs with T_max=300** (v45): 3.02. Proper LR schedule (T_max matching epoch count) helps slightly over just running the same arch past its original T_max. But MoG (below) is a bigger lever.

## Phase 5: Mixture of Gaussians - the next breakthrough (May 6 afternoon)

The single-Gaussian NLL model predicts one mean + variance per player per timestep. The fundamental problem: basketball is multimodal. A player driving toward a defender can go left *or* right — the correct prediction is not the average of those two paths (which would land in the middle, through the defender). A single Gaussian predicts the mean; a mixture predicts the distribution.

Changed the output head to predict K=2 Gaussian components per player per timestep: K×5 outputs (mu_x, mu_y, log_sigma_x, log_sigma_y, logit_weight). Training loss is the log-likelihood of the mixture (logsumexp over components). At inference, returns the weighted mean of component means.

- **MoG K=2 at 100 epochs** (v44x100): 3.12. Beats single-Gaussian at same epochs (v43x100=3.19) by 3.9%.
- **MoG K=2 at 250 epochs** (v44x250): **2.90. Massive new best. 4.8% improvement over v43×250ep (3.04).** This is the largest single-step gain since the NLL breakthrough itself.

The pattern is clear: MoG captures something the single Gaussian physically cannot — the bimodal future distribution of players facing a decision point. And it compounds with epoch scaling just like every other architectural improvement we've found.

## Current best results

| Model | Local MSE | Kaggle MSE | Notes |
|-------|-----------|------------|-------|
| **v44 × 250ep TTA** | **2.8963** | **pending** | MoG K=2, new best |
| v43 × 250ep TTA | 3.0395 | pending | K=2 + timestep-weighted NLL |
| v45 × 300ep TTA | 3.0177 | pending | v42 arch, proper T_max=300 |
| v42 × 250ep | 3.0558 | — | K=2 nearest + NLL |
| v42 × 100ep TTA | 3.2099 | ~3.44785 | best Kaggle so far |
| v38 TTA | 3.4355 | 3.68213 | GRU + NLL baseline |

## Key lessons

1. **NLL loss is a huge improvement over MSE** for this task. Even at 35 epochs, NLL beats anything MSE could do at 200+ epochs.
2. **Mixture of Gaussians (MoG) is the next big win after NLL.** Basketball is multimodal — players face discrete decisions (left or right, pass or drive). A single Gaussian predicts the average of those futures; MoG predicts the actual distribution. K=2 components gave the biggest single-step gain since switching to NLL.
3. **Epoch scaling beats feature engineering**. The biggest single gains came from training longer, not from adding features.
4. **Social features (K=2 nearest) matter more at longer training**. They barely help at 35ep but unlock significant gains at 100ep.
5. **Timestep weighting is epoch-dependent**: hurts at 35ep, helps at 100ep and beyond. The model needs enough training capacity to act on the extra gradient signal for far-future steps.
6. **Bigger models don't help**. The 128d 2-layer GRU consistently beats 192d 3-layer at the same epoch budget. The problem doesn't need more capacity.
7. **Dropout kills autoregressive models**. The GRU hidden state carries trajectory history autoregressively - dropout corrupts it badly.
8. **Directional attention is fundamentally broken for autoregressive decoding** because predicted positions can converge (dist->0), causing gradient explosion. Stick to scalar distance bias.
9. **Court symmetry augmentation** is free +1-2% — always use it.
10. **LR schedule must match epoch count**: CosineAnnealingLR reaches minimum at T_max; running past T_max means training at near-zero LR, which wastes compute. Always set T_max=epochs.

## What to try next

**Highest priority:**
- **v44 × 300ep** (MoG K=2, T_max=300): epoch scaling has improved every single model we've run longer; v44 at 250ep is 2.8963, estimate ~2.85-2.87 at 300ep.
- **Ensemble TTA: v43×250ep + v44×250ep**: average predictions from both TTA CSVs. Two complementary uncertainty models (Gaussian vs MoG) should cancel each other's errors. Free gain, no training needed.

**Architectural exploration:**
- **MoG K=3**: more components for richer multimodal modeling. K=2 captures binary decisions (left/right); K=3 could capture three-way choices (drive left, drive right, pass back). Low risk, moderate expected gain.
- **Cross-model ensemble**: v43 (Gaussian) + v44 (MoG) differ in their uncertainty model, so their residual errors should be less correlated than two runs of the same arch.
