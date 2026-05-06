# Competition Notes

## Overfitting signal: watch the local→Kaggle gap

Every time we submit to Kaggle, compare:

| Submission | Local | Kaggle | Gap |
|------------|-------|--------|-----|
| v38 TTA | 3.4355 | 3.6821 | +7.2% |
| v42x100ep TTA | 3.2099 | 3.4478 | +7.4% |

The gap is stable at ~7%. This is healthy — test set is slightly harder than our val set, but consistently so.

**Red flag**: if gap grows to 11%+ on a future submission, stop trusting local scores.
Three causes: (1) val set overfitting from 40+ experiments on same 994 files, (2) public LB overfitting from repeated submissions, (3) train overfitting (watch train loss vs val loss in logs).

As long as gap stays 7-8%, local improvements are trustworthy and we should keep optimizing locally.

## Val set: is it worth changing?

Current split: 3974 train / 994 val (fixed, from train_files.txt / val_files.txt).

**Short answer: no, not worth changing now.**

- Our 994-file val set is clearly representative — the 7% gap is stable across multiple model generations
- K-fold CV would be 5x slower and not change the architecture decisions we'd make
- Retraining everything on a new split costs GPU time with no clear gain

**What would be worth it**: creating a small held-out "probe" set of ~200 files that we never use for architecture selection — only to check if val score is still honest. Pull them out of current val set. But given we're #1 and the gap is stable, this is low priority.

## Hyperparameter tuning: is it worth it?

Things we've implicitly tuned already through experiments:
- lr=1e-3 (tried 5e-4, worse)
- CosineAnnealingLR with T_max=epochs (tried other values, worse)
- batch_size=32
- state_dim=128, 2 layers (tried bigger, worse)
- weight_decay=5e-4 (never explicitly tuned this one)

**Probably not worth a dedicated HP search.** The model is not very sensitive to these — the big gains came from architecture changes (NLL loss, K=2 social features) and epoch scaling, not HP tuning. The one thing untested is weight_decay (try 1e-4 or 1e-3 sometime if a slot is free).

## What actually moves the needle (in order of impact)

1. Loss function: NLL > MSE by a huge margin
2. Epoch scaling: 35ep → 100ep → 250ep
3. Social features: K=2 nearest neighbors
4. Court augmentation: x/y flip
5. Everything else: marginal
