# Plan: attack the ball error

Built on the finding in [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md): the **ball (entity 0) is ~39% of
total error at 6.3× a player's MSE**. Closing that gap is worth far more than any player-side tweak
(ball→avg-player accuracy = local 2.82 → 1.90). This plan goes cheapest-first, reuses the existing
v46 backbone and the per-entity ensemble pipeline, and measures every step with
`misc/analyze_error_sources.py` (track **ball MSE**, not just overall).

**Compounding bonus:** players consume the broadcast ball position during rollout
(`_ball_broadcast` → `motion` in `nba_v46.py`). So early-step ball accuracy feeds back into player
accuracy too — the ball lever is partly a player lever.

**Baseline to beat (v46_300ep, TTA):** overall 2.818 · ball 12.04 · players 1.896.

---

## Phase 0 — Instrumentation (done / keep using)
- `misc/analyze_error_sources.py [--tta]` reports overall + ball + player + per-timestep MSE.
- **Acceptance rule for every experiment:** report overall AND ball AND player MSE. A change is a
  win only if overall drops; watch that ball gains aren't fully eaten by player regression (the
  metric is uniform, so the trade is real).

## Phase 1 — Ball-weighted loss  (cheapest; no architecture change)  → `nba_v60.py`
The `MoGNLLLoss` in `nba_v46.py` weights timesteps (`linspace 1→2`) but treats all 11 entities
equally. Add a **per-entity weight** that upweights the ball.

- In `MoGNLLLoss.compute`, the per-element negative log-mix is `-log_mix[t]` of shape `[B*N]`.
  Build an entity weight vector `w_ent` (length N, ball index 0 = `BALL_W`, players = 1.0),
  broadcast over the batch, and weight the mean: `(w_ent * -log_mix[t]).sum() / w_ent.sum()`
  instead of `-log_mix[t].mean()`.
- Ball index is always 0, so `w_ent = [BALL_W, 1,1,...,1]` tiled per batch element.
- **Sweep `BALL_W ∈ {2, 3, 5, 8}`** as v60_w2 … v60_w8. Everything else identical to v46
  (300ep, T_max=300, seed 0) for a clean A/B.
- Expected: ball MSE down, player MSE up slightly; find the `BALL_W` that minimizes overall.
- If even `BALL_W=2` raises overall, the trunk is capacity-bound on the ball → go to Phase 3.

## Phase 2 — More mixture modes for the ball  → `nba_v61.py`
Global K=3 didn't help (players don't need it; v47/v54). But the ball is genuinely multimodal
(pass / shot / dribble). Give the **ball more components without touching players**.

- Keep player head at K=2. Add a **separate ball head** with `K_BALL ∈ {3, 4}` (and optionally
  heavier-tailed — Laplace — for the ball only, since global Laplace regressed *because of players*).
- Route entity 0's trunk state `h[-1]` through `proj_ball`, entities 1–10 through `proj` (current).
- Combine in the existing eval-mode mixture-mean reduction, per-entity K.
- Compare against the best Phase 1 model.

## Phase 3 — Ball-specialist head on a shared trunk  → `nba_v62.py`
If Phase 1/2 show the shared head is the bottleneck, widen the ball path.
- Shared GRU+attn trunk (unchanged). Two heads: `proj_player` (current 128→64→K·5) and a **wider
  `proj_ball`** (e.g. 128→128→64→K_BALL·5) selected by entity index.
- Combine Phase 1's ball-weighted loss + Phase 2's ball K. This is the "all ball levers on the
  shared trunk" model.
- Alternative if this plateaus: a **fully separate ball-only model** (trained only on entity 0,
  possibly with richer ball features) ensembled in via Phase 5 — the per-entity weights already
  let entity 0 select it alone.

## Phase 4 — Ball-aware input features  → `nba_v63.py`
The ball's own prediction currently sees the same motion/social features as players. Add cues that
explain pass/shot onset (data is 2D: cols = x, y, ball-flag, team — no height):
- **Possession proxy:** distance from ball to nearest player; identity/team of nearest player.
- **Pass/shot onset:** ball speed magnitude and its change over the context window (the trunk has
  velocity but not an explicit "ball is accelerating hard" signal).
- **Time-aligned:** compute per context frame; feed only into entity 0's input (or to all, since
  players already get broadcast ball pos).
- Cheap to try inside `_social_features` / `motion` construction; keep a v46 control.

## Phase 5 — Fold into the per-entity ensemble  (final)
The submission path already fits **one weight vector per entity**
(`misc/weighted_ensemble_per_entity.py`), so a ball-specialist is naturally selected for entity 0
only and ignored for players.
- Run `save_val_tta_preds.py` + `save_test_tta_preds.py` for the best ball model(s).
- Refit per-entity weights over {existing 21-model pool + new ball models}.
- Success = the per-entity ensemble's overall val MSE drops below the current 2.7799, driven by
  entity-0 weight mass moving onto the ball specialists.

---

## Order of operations & exit criteria
1. **Phase 1** first — one file, one loss edit, a 4-point sweep. If `BALL_W` finds a lower overall
   MSE, ship it into the ensemble (Phase 5) immediately as a quick win.
2. **Phase 2** next — highest expected ball-specific gain for moderate effort.
3. **Phase 3/4** only if 1–2 leave ball MSE >> player MSE (they will; aim to shrink the 6.3× gap).
4. Re-run `analyze_error_sources.py` after each; log ball MSE alongside overall in `results.md`.

**Definition of done for this thrust:** ball MSE materially below 12.0 with overall val improved,
and the per-entity ensemble crossing under 2.78 local (→ sub-3.0 Kaggle with margin).
