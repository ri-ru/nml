# Error Source Analysis — the ball dominates

**TL;DR:** The ball (entity 0) produces **~39% of the total validation error while being only
1 of 11 entities (9.1%)**. Its MSE is **~6.3× that of any player**. If the ball were predicted
as accurately as an average player, local MSE would drop from **2.82 → 1.90 (−32.7%)**. The ball
is the single highest-leverage target in this competition.

Reproduce: `python misc/analyze_error_sources.py [--tta]`
(decomposes the same denormalized MSE that `evaluate.py` reports, against `model_v46_300ep.pth`).

---

## How the score is computed

`evaluate.py` computes MSE **uniformly over all 11 entities** (10 players + 1 ball), all 12
timesteps, both axes. The ball is therefore weighted at exactly 1/11 of the averaged elements —
but it contributes far more than 1/11 of the error.

The ball is **always entity index 0** (verified across sequences: `seq[0,:,2]==0` marks it, and the
pattern is `(0,)` in every file). So error can be split cleanly by entity index.

## Ball vs players (v46_300ep, 4-flip TTA)

| Group | MSE | Share of total error |
|---|---|---|
| **Ball** (1 entity) | **12.04** | **38.8%** |
| Players (10 entities) | 1.90 | 61.2% |

- Ball MSE / player MSE = **6.35×**
- Players are tightly clustered (1.71–2.08); the ball is a lone outlier at 12.04.
- **Counterfactual:** ball at average-player accuracy → overall MSE **1.896** (−32.7%). This is
  larger than the cumulative gain from every architecture change logged in `findings.md`.

Per-entity MSE (entity 0 = ball):

```
  entity  0: 12.0397   <-- BALL
  entity  1:  1.7092
  entity  2:  1.8436
  entity  3:  1.8008
  entity  4:  1.8569
  entity  5:  1.9159
  entity  6:  2.0236
  entity  7:  1.9085
  entity  8:  1.8954
  entity  9:  2.0768
  entity 10:  1.9275
```

## Per-timestep: the ball is worst *early* (relatively)

| t | overall | ball | players | ball share of that step's error |
|---|---|---|---|---|
| 1 | 0.069 | 0.438 | 0.032 | **57.8%** |
| 2 | 0.130 | 1.055 | 0.038 | **73.8%** |
| 3 | 0.275 | 1.940 | 0.108 | 64.2% |
| 4 | 0.554 | 3.610 | 0.248 | 59.3% |
| 5 | 0.944 | 5.451 | 0.493 | 52.5% |
| 6 | 1.491 | 8.020 | 0.838 | 48.9% |
| 7 | 2.191 | 11.093 | 1.301 | 46.0% |
| 8 | 3.068 | 14.603 | 1.914 | 43.3% |
| 9 | 4.144 | 18.532 | 2.705 | 40.7% |
| 10 | 5.427 | 22.652 | 3.705 | 37.9% |
| 11 | 6.913 | 26.660 | 4.938 | 35.1% |
| 12 | 8.612 | 30.423 | 6.431 | 32.1% |

- At t=1–2 the ball is **58–74% of the entire error**: players move smoothly and the model nails
  their near future (player MSE 0.032 at t=1), but the ball gets **passed and shot** — abrupt
  velocity changes the 8-frame context can't telegraph.
- As the horizon extends, player error compounds through autoregressive rollout, so the ball's
  *share* falls — but its *absolute* error keeps leading (30.4 vs 6.4 at t=12).

## Axis

Symmetric: x = 2.87, y = 2.77. No court-orientation bias — flip augmentation/TTA is doing its job.

---

## Implications — where to spend effort next

Nothing player-side will move the needle much. The headroom is the ball:

1. **Ball-weighted loss.** Upweight entity 0 in the training MoG NLL. The model currently spends
   equal capacity on 10 easy entities and 1 hard one. Cheapest test, directly attacks the 33%.
2. **Ball-specialist head / model.** The ball has different dynamics (ballistic arcs, possession
   changes) than players — a dedicated head or a separate model for entity 0, ensembled in.
3. **Ball-aware features.** Players already receive broadcast ball position; the *ball's own*
   prediction may benefit from explicit possession / nearest-player features and pass/shot-onset cues.
4. **Heavier-tailed / more multimodal output for the ball only.** Laplace MoG regressed *globally*
   (2.94) because players don't need heavy tails — but the ball's discrete pass/shoot modes are
   exactly where extra mixture components or heavier tails should pay off.

This is also why **per-entity ensemble weights** (`weighted_ensemble_per_entity.py`) help: they let
the ball pick a different model subset than the players instead of forcing one compromise vector.
