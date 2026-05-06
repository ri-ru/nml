# Run Log

Baseline GRU : each player predicted independently, no interaction, no velocity. Score: 54.0

GAT : added graph attention so players can see each other. Score: 23.3

dist:GAT : made attention distance:biased (closer = more attention). Score: 9.3

v3 : added explicit velocity (vx, vy) as input features, autoregressive decoding. Score: 6.0

v4 : tried non:autoregressive: predict constant:velocity baseline + GRU correction all at once. Score: 6.6. Worse.

v5 : scheduled teacher forcing + bigger model (192d, 3 layers). Score: 11.9. Teacher forcing caused train/inference gap, killed it.

v6 : added ball:relative features (where is each player relative to the ball). Score: 4.36

v7 : replaced GRU context encoder with Transformer. Score: 4.63. No gain, dropped it.

v8 : tried predicting displacements instead of absolute positions. Score: 5.39. Error accumulates, worse.

v9 : constant:velocity prior + GRU residual + bigger model. Score: 5.02. Still worse than v6.

v10 : added team centroid offsets + ball:direction cosine. Score: 4.37. No improvement.

v11 : halved learning rate (5e:4 instead of 1e:3). Score: 4.77. Too slow.

v12 : added nearest teammate + nearest opponent relative pos/vel. Score: 4.16

v13 : threw everything in together (v6 + v12 + v10). Score: 4.62. More features hurt.

v14 : directional attention (dx, dy vectors instead of scalar distance). NaN at epoch 3. Gradient explosion when players get close. Dead end, tried twice (v16 too), always NaN.

v15 : K=2 nearest teammate + K=2 nearest opponent (16 social features). Score: 4.04

v17 : K=3 nearest per team. Score: 4.09. Worse than K=2, third neighbor is noise.

v18 : added context summary (mean velocity + last velocity from 8 frames) as decoder input. Score: 4.08. GRU already captures this.

v19 : v15 arch, trained 100 epochs instead of 35. Score: 3.84. Epoch scaling > feature engineering.

v20 : K=1 hard nearest + soft distance:weighted attention over all teammates. Score: 4.03.

v21 : GNN edge:MLP message passing instead of typed attention. Score: 4.12. Worse at 35ep.

v22 : decoder self:attention over previous steps. Score: 4.35. GRU already does this implicitly.

v23 : v15 arch, 150 epochs. Score: 3.77

v24 : v20 arch (soft global attention), 100 epochs. Score: 3.83

v26 : v24 + court symmetry augmentation (random x:flip, y:flip). Score: 3.79

v27 : scaled model (256d, 3 layers), 35 epochs. Score: 6.94. Late bloomer, would need 150+ epochs.

v28 : added acceleration features. Score: 4.09. GRU already differentiates velocity implicitly.

v29 : longer context (16 frames instead of 8). Score: 5.53. Fewer training sequences, worse.

v30 : v26 arch (aug + v24), 150 epochs. Score: 3.75

v31 : v26 arch, 200 epochs. Score: 3.73

v32 : v26 + dropout 0.15, 150 epochs. Score: 4.82. Dropout destroys autoregressive GRU hidden state.

v33 : v26 arch, 250 epochs. Score: 3.75. Plateau, slightly worse than 200ep.

v37 : non:autoregressive Transformer + Gaussian NLL loss (first NLL experiment). Score: 3.86

v38 : GRU + Gaussian NLL, 35 epochs. Score: 3.44. !!!! : NLL beats everything from before. Kaggle: 3.68 with TTA.

v39 : bigger GRU (192d, 3 layers) + NLL, 35 epochs. Score: 6.37 (late bloomer). Ran to 150 epochs: 3.85. Still worse than v38 at 35ep.

v40 : Transformer + NLL, 35 epochs. Score: 3.92. Ran to 100 epochs: 3.42.

v41 : v38 arch + timestep:weighted NLL (weights 1.0 to 2.0 across the 12 steps). Score: 3.43. No real gain over uniform NLL.

v42 : v38 arch + K=2 nearest neighbors + NLL, 35 epochs. Score: 3.43. Marginal. Ran to 100 epochs: 3.21. K=2 social features only shine with enough training.

v43 : v42 arch + timestep:weighted NLL, 35 epochs. Score: 3.45. Worse. Running to 100 epochs now.

v42 x100ep : Score: 3.21. Big gain. K=2 social + epoch scaling compound.

v43 x100ep : Score: 3.19. New best. Timestep weighting helps at 100ep even though it hurt at 35ep.

v42 x250ep : Score: 3.06. Epoch scaling continues. New best.

v43 x250ep : Score: 3.04. New best. Timestep weighting keeps helping at 250ep.

v44 : MoG K=2 output on v43 arch. Instead of predicting one Gaussian per player, predict a mixture of 2. Loss is log-likelihood of mixture. Inference uses weighted mean of component means. 100 epochs: 3.12. Promising.

v44 x250ep : Score: 2.90. Massive jump. MoG captures bimodal futures better than a single Gaussian. New best by 4.8%.

v45 : v42 arch retrained from scratch for 300 epochs with T_max=300. Hypothesis: proper LR schedule helps over just running v42 past 250ep. Score: 3.02. Beats v43x250ep but trails v44. MoG is the better bet.
