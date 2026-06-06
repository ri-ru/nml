# Deep Ball-Error Analysis (model=v60_w8, baseline=v46)
N windows: 2982

Mean ball MSE: v60_w8=11.623  v46=12.040

## A. Concentration (is it a heavy tail?)
- top 305 windows (10.2%) = 50% of ball error
- top 1013 windows (34.0%) = 80% of ball error
- top 1493 windows (50.1%) = 90% of ball error
- median ball MSE 5.463 vs mean 11.623 (mean>>median => heavy tail)
- worst 1% mean 156.1; best 50% mean 2.332

## B. Ball error vs context speed (last-step |v|, feet/frame)
- speed [0.00,0.90) n= 746  ball MSE 12.966
- speed [0.90,1.79) n= 745  ball MSE 12.864
- speed [1.79,2.19) n= 745  ball MSE 9.848
- speed [2.19,32.02) n= 746  ball MSE 10.813

## C. Model vs constant-velocity ball baseline
- CV ball MSE 70.265  vs model 11.623 (model better)
- model beats CV on 87.2% of windows; CV beats model on 12.8%
- on windows where CV wins: mean speed 0.91 vs overall 1.61

## D. Hedging (does the model under-move the ball?)
- mean predicted net displacement 7.58 vs actual 10.14 (ratio 0.75; <1 = under-moves)
- on big-move balls (top 25% actual disp): pred 13.96 vs actual 19.65 (ratio 0.71)

## E. Ball error vs nearest-player distance at present (possession proxy)
- dist [0.1,1.5) n= 746  ball MSE 15.460
- dist [1.5,2.4) n= 745  ball MSE 11.863
- dist [2.4,4.6) n= 743  ball MSE 9.426
- dist [4.6,13.2) n= 748  ball MSE 9.740

## F. Per-timestep ball MSE
    t    v60_w8       v46        CV
    1     0.257     0.438     1.016
    2     0.941     1.055     3.285
    3     1.705     1.940     7.059
    4     3.308     3.610    13.794
    5     5.109     5.451    23.574
    6     7.628     8.019    37.635
    7    10.665    11.093    56.173
    8    14.143    14.603    79.071
    9    18.014    18.532   106.251
   10    22.048    22.652   137.010
   11    25.982    26.660   170.817
   12    29.678    30.423   207.495
