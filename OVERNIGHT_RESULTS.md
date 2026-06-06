# Overnight Ensemble Results
Y_real: (2982, 12, 11, 2)

## base  (21 models)
- models: ['v46', 'v58', 'v46_s1', 'v46_s2', 'v46_s3', 'v46_s4', 'v46_s5', 'v46_s6', 'v46_s7', 'v46_s8', 'v46_s1_swa', 'v46_s2_swa', 'v46_s3_swa', 'v46_s4_swa', 'v46_s5_swa', 'v46_s6_swa', 'v46_s7_swa', 'v46_s8_swa', 'v58_s1', 'v58_s2', 'v58_s3']
- **cons**: overall 2.7865  ball 11.916  -> `solution_overnight_base_cons.csv`
- **per-entity**: overall 2.7799  ball 11.896  -> `solution_overnight_base_perentity.csv`
- phys 2.7799->2.7799 (no)

## base+ball  (27 models)
- models: ['v46', 'v58', 'v46_s1', 'v46_s2', 'v46_s3', 'v46_s4', 'v46_s5', 'v46_s6', 'v46_s7', 'v46_s8', 'v46_s1_swa', 'v46_s2_swa', 'v46_s3_swa', 'v46_s4_swa', 'v46_s5_swa', 'v46_s6_swa', 'v46_s7_swa', 'v46_s8_swa', 'v58_s1', 'v58_s2', 'v58_s3', 'v60_w5', 'v60_w8', 'v60_w3', 'v60_w8_s1', 'v60_w8_s2', 'v60_w5_s1']
- **cons**: overall 2.7833  ball 11.817  -> `solution_overnight_base+ball_cons.csv`
- **per-entity**: overall 2.7525  ball 11.595  -> `solution_overnight_base+ball_perentity.csv`
- phys 2.7525->2.7525 (no)

## base+ball+pseudo  (28 models)
- models: ['v46', 'v58', 'v46_s1', 'v46_s2', 'v46_s3', 'v46_s4', 'v46_s5', 'v46_s6', 'v46_s7', 'v46_s8', 'v46_s1_swa', 'v46_s2_swa', 'v46_s3_swa', 'v46_s4_swa', 'v46_s5_swa', 'v46_s6_swa', 'v46_s7_swa', 'v46_s8_swa', 'v58_s1', 'v58_s2', 'v58_s3', 'v60_w5', 'v60_w8', 'v60_w3', 'v60_w8_s1', 'v60_w8_s2', 'v60_w5_s1', 'v68']
- **cons**: overall 2.7833  ball 11.817  -> `solution_overnight_base+ball+pseudo_cons.csv`
- **per-entity**: overall 2.7522  ball 11.595  -> `solution_overnight_base+ball+pseudo_perentity.csv`
- phys 2.7522->2.7522 (no)

## base+ball+agent  (29 models)
- models: ['v46', 'v58', 'v46_s1', 'v46_s2', 'v46_s3', 'v46_s4', 'v46_s5', 'v46_s6', 'v46_s7', 'v46_s8', 'v46_s1_swa', 'v46_s2_swa', 'v46_s3_swa', 'v46_s4_swa', 'v46_s5_swa', 'v46_s6_swa', 'v46_s7_swa', 'v46_s8_swa', 'v58_s1', 'v58_s2', 'v58_s3', 'v60_w5', 'v60_w8', 'v60_w3', 'v60_w8_s1', 'v60_w8_s2', 'v60_w5_s1', 'v65', 'v66']
- **cons**: overall 2.7833  ball 11.817  -> `solution_overnight_base+ball+agent_cons.csv`
- **per-entity**: overall 2.7521  ball 11.595  -> `solution_overnight_base+ball+agent_perentity.csv`
- phys 2.7521->2.7521 (no)

## all  (30 models)
- models: ['v46', 'v58', 'v46_s1', 'v46_s2', 'v46_s3', 'v46_s4', 'v46_s5', 'v46_s6', 'v46_s7', 'v46_s8', 'v46_s1_swa', 'v46_s2_swa', 'v46_s3_swa', 'v46_s4_swa', 'v46_s5_swa', 'v46_s6_swa', 'v46_s7_swa', 'v46_s8_swa', 'v58_s1', 'v58_s2', 'v58_s3', 'v60_w5', 'v60_w8', 'v60_w3', 'v60_w8_s1', 'v60_w8_s2', 'v60_w5_s1', 'v65', 'v66', 'v68']
- **cons**: overall 2.7833  ball 11.817  -> `solution_overnight_all_cons.csv`
- **per-entity**: overall 2.7519  ball 11.595  -> `solution_overnight_all_perentity.csv`
- phys 2.7519->2.7519 (no)

## Caveat
Per-entity OVERFITS val (11xM params). Local overall is optimistic; trust the gap less for per-entity. cons is the robust pick. Compare ball MSE vs the prior best ensemble ball=11.896.
