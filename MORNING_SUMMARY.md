
## 100% candidate
100%-DATA CANDIDATE: solution_full_2grp.csv
  0/27 members use _full (100%-data) test preds; rest 80%.
  reference local 2-group (80% preds) = 2.7563; submitted 2-group was 2.7563 -> Kaggle 2.98086.
  NOTE: local can't measure the _full gain (val is in their training set). Submit and compare on Kaggle.

# MORNING SUMMARY (autonomous run 2026-06-07 00:37)

- 100% retrain complete: 14/14 _full weights.


## 100% candidate
100%-DATA CANDIDATE: solution_100pct_full_2grp_NEW.csv
  14/27 members use _full (100%-data) test preds; rest 80%.
  reference local 2-group (80% preds) = 2.7563; submitted 2-group was 2.7563 -> Kaggle 2.98086.
  NOTE: local can't measure the _full gain (val is in their training set). Submit and compare on Kaggle.

## GNN-diversity ensemble sweep (see OVERNIGHT_RESULTS.md)

  ## base+ball  (27 models)

  - **cons**: overall 2.7833  ball 11.817  -> `solution_overnight_base+ball_cons.csv`

  - **per-entity**: overall 2.7525  ball 11.595  -> `solution_overnight_base+ball_perentity.csv`

  ## base+ball+pseudo  (28 models)

  - **cons**: overall 2.7833  ball 11.817  -> `solution_overnight_base+ball+pseudo_cons.csv`

  - **per-entity**: overall 2.7522  ball 11.595  -> `solution_overnight_base+ball+pseudo_perentity.csv`

  ## base+ball+gnn  (29 models)

  - **cons**: overall 2.7828  ball 11.820  -> `solution_overnight_base+ball+gnn_cons.csv`

  - **per-entity**: overall 2.7519  ball 11.595  -> `solution_overnight_base+ball+gnn_perentity.csv`


Candidates ready: solution_full_2grp.csv (100% data) ; solution_overnight_base+ball+gnn_*.csv (GNN diversity). Submit & compare on Kaggle vs current best 2.98086.

