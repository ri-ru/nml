# Autonomous overnight runner (per user). Runs AFTER the 100% retrain queue finishes; GPU-idle-gated.
# P1 wait for retrains -> P2 build 100%-data candidate -> P3 GNN-diversity ensemble test.
# Everything logged to MORNING_SUMMARY.md. Deadline-bounded so it never runs away.
$ErrorActionPreference = 'Continue'
$proj = "C:\Users\jan\Downloads\nml_project_proposal\nml_project_proposal"
$py   = "C:\Users\jan\Downloads\nml_project_proposal\.venv\Scripts\python.exe"
$log  = "$proj\models\logs\overnight_auto.log"
$sum  = "$proj\MORNING_SUMMARY.md"
$deadline = (Get-Date).AddHours(12)
function Say($m){ $t="[$(Get-Date -Format 'HH:mm:ss')] $m"; Add-Content $log $t; Write-Output $t }
function Idle(){ return (Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match 'run_one|retrain_full' }).Count -eq 0 }

Add-Content $sum "`n# MORNING SUMMARY (autonomous run $(Get-Date -Format 'yyyy-MM-dd HH:mm'))`n"
Say "=== AUTONOMOUS RUNNER START ==="

# ---- P1: wait for the 100% retrain queue to finish ----
Say "P1: waiting for '100PCT RETRAIN DONE' in retrain100.log..."
while ((Get-Date) -lt $deadline) {
  if (Select-String -Path "$proj\models\logs\retrain100.log" -Pattern "100PCT RETRAIN DONE" -EA SilentlyContinue) { break }
  Start-Sleep -Seconds 120
}
$nfull = (Get-ChildItem "$proj\models\weights\model_*_full.pth" -EA SilentlyContinue | Measure-Object).Count
Say "P1 done. $nfull _full weights present."
Add-Content $sum "- 100% retrain complete: $nfull/14 _full weights.`n"

# ---- P2: build the 100%-data candidate ----
Say "P2: computing _full test preds + building 100% candidate..."
& $py "$proj\misc\save_test_full.py"     2>&1 | Add-Content $log
& $py "$proj\misc\build_full_candidate.py" 2>&1 | Add-Content $log
Say "P2 done -> solution_full_2grp.csv"

# ---- P3: GNN-diversity ensemble test (does a genuinely different arch help?) ----
if ((Get-Date) -lt $deadline) {
  Say "P3: GNN diversity. Saving v72/v72_w8 preds + ensemble sweep..."
  & $py "$proj\misc\save_val_tta_preds.py"  2>&1 | Add-Content $log
  & $py "$proj\misc\save_test_tta_preds.py" 2>&1 | Add-Content $log
  & $py "$proj\misc\overnight_ensemble.py"  2>&1 | Add-Content $log
  $r = Get-Content "$proj\OVERNIGHT_RESULTS.md" -EA SilentlyContinue | Select-String -Pattern "base\+ball|gnn"
  Add-Content $sum "`n## GNN-diversity ensemble sweep (see OVERNIGHT_RESULTS.md)`n"
  $r | ForEach-Object { Add-Content $sum ("  " + $_.Line + "`n") }
  Say "P3 done."
}

Say "=== AUTONOMOUS RUNNER DONE. See MORNING_SUMMARY.md + models/solutions/solution_full_2grp.csv ==="
Add-Content $sum "`nCandidates ready: solution_full_2grp.csv (100% data) ; solution_overnight_base+ball+gnn_*.csv (GNN diversity). Submit & compare on Kaggle vs current best 2.98086.`n"
