# NBA trajectory forecasting

Predict the next **12 positions of all 10 players and the ball** from eight observed frames. The model combines spatio-temporal graph attention, a recurrent rollout, and a mixture-of-Gaussians prediction head.

**Kaggle submission score: 2.954**, using six models of one architecture with separate ensemble weights for the ball and players. The repository's scorer uses mean-squared error in court coordinates (feet²), averaged over entities, timesteps, and axes; lower is better.

[Model walkthrough](NML_FINAL.md) · [Architecture](demarche/nba_v46.py) · [Ensemble configuration](final/ensemble_weights.json) · [Submitted CSV](outputs/solutions/solution_ensemble_6model.csv)

## The useful insight: the ball dominates the error

Error analysis found that the ball accounted for approximately **39% of total prediction error**, despite being only one of 11 entities. Treating every entity identically left performance on the table.

We trained three ball specialists with higher ball-loss weights and three player generalists, then fitted two constrained ensemble-weight vectors on a held-out 20% split. Submission members were retrained on all labelled data, keeping validation-weight fitting separate from the final training run.

## Architecture and iteration

| Stage | Change |
|---|---|
| Independent GRU baseline | Model each entity without graph interactions |
| Graph attention | Add ball-relative motion and typed pairwise interactions |
| Gaussian prediction | Learn a distribution rather than only a point estimate |
| Mixture prediction | Two-component Gaussian head with horizon-weighted NLL |
| Error-driven ensemble | Ball-weighted specialists, player generalists, and four-flip test-time augmentation |

Graph attention uses distance and entity/team-type biases. A per-node GRU carries temporal state; the future rollout consumes the model's previous prediction. All six final members share the same core architecture.

## Run

Install the dependencies and point `NML_DATA` at the dataset directory containing `train/train/*.pt` and `test/test/*.pt`:

```bash
python -m pip install -r requirements.txt
export NML_DATA=/path/to/nml-2026
python run.py
```

Rebuild predictions from the committed full-data checkpoints:

```bash
python run.py submit
python run.py submit-single v46
```

Train a member or run the full training pipeline:

```bash
python run.py train v46 300
python run.py reproduce
```

Prediction and training require the dataset and a suitable PyTorch environment; a CUDA GPU is recommended. `fit-weights` prints fitted weights for inspection: update `final/ensemble_weights.json` to use newly fitted values. The supplied configuration preserves the final submission's recorded weights.

## Code map

- [run.py](run.py): launcher for splitting, training, fitting, and prediction.
- [demarche/](demarche/): architecture milestones and ensemble-member training.
- [utils/](utils/): shared split, training, and scoring utilities.
- [final/](final/): single-model and six-model prediction builders.
- [outputs/](outputs/): trained checkpoints and submitted prediction files.

Built with teammates; see the [contributor history](https://github.com/ri-ru/nml/graphs/contributors). The [final walkthrough](NML_FINAL.md) and ensemble configuration record the score and method; no leaderboard rank is claimed.
