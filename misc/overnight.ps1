# Overnight autonomous pipeline for the NBA project.
# Phase 1: wait for v60 ball runs -> save preds -> lock incremental ensemble + round-2 pseudo
# Phase 2: train v65/v66 (agent-centric, 300ep) + v68 (pseudo round-2, 300ep)
# Phase 3: final ensembling across everything -> candidate CSVs + OVERNIGHT_RESULTS.md
$ErrorActionPreference = 'Continue'
$proj = "C:\Users\jan\Downloads\nml_project_proposal\nml_project_proposal"
$py   = "C:\Users\jan\Downloads\nml_project_proposal\.venv\Scripts\python.exe"
$log  = "$proj\models\logs\overnight.log"
function Say($m){ $t=(Get-Date).ToString('HH:mm:ss'); Add-Content $log "[$t] $m"; Write-Output "[$t] $m" }

function WaitScores($tags, $maxMin){
  $deadline = (Get-Date).AddMinutes($maxMin)
  while ((Get-Date) -lt $deadline){
    $allDone = $true
    foreach($tag in $tags){
      $f = "$proj\models\logs\$tag.log"
      if (-not (Test-Path $f) -or -not (Select-String -Path $f -Pattern "AGENT_SCORE" -EA SilentlyContinue)){ $allDone=$false }
    }
    if ($allDone){ return $true }
    Start-Sleep -Seconds 60
  }
  return $false
}

Say "=== OVERNIGHT START ==="

# ---------- Phase 1: wait for v60 300ep ----------
Say "Phase 1: waiting for v60_w5/w8/w3 300ep..."
WaitScores @('v60_w5_300ep_train','v60_w8_300ep_train','v60_w3_300ep_train') 150 | Out-Null
Say "v60 done (or timed out). Saving TTA preds (v60 ball models)..."
& $py "$proj\misc\save_val_tta_preds.py"  2>&1 | Add-Content $log
& $py "$proj\misc\save_test_tta_preds.py" 2>&1 | Add-Content $log
Say "Running ensemble sweep (incremental, v60 folded)..."
& $py "$proj\misc\overnight_ensemble.py" 2>&1 | Add-Content $log

# round-2 pseudo from the best new ensemble CSV (fallbacks handled by v68 itself)
$bestcsv = "$proj\models\solutions\solution_overnight_base+ball_cons.csv"
if (Test-Path $bestcsv){
  Say "Generating round-2 pseudo from $([IO.Path]::GetFileName($bestcsv))..."
  & $py "$proj\misc\make_pseudo_generic.py" "$bestcsv" 2>&1 | Add-Content $log
} else {
  Say "No base+ball CSV; v68 will fall back to existing pseudo set."
}

# ---------- Phase 2: agent-centric + pseudo at 300ep ----------
Say "Phase 2: launching v65 + v66 + v68 at 300ep..."
foreach($m in @('v65','v66','v68')){
  Start-Process -WindowStyle Hidden -FilePath $py `
    -ArgumentList "$proj\run_one.py $proj\models\nba_$m.py 300 32" `
    -RedirectStandardOutput "$proj\models\logs\${m}_300ep_train.log" `
    -RedirectStandardError  "$proj\models\logs\${m}_300ep_err.log" `
    -WorkingDirectory $proj
  Say "launched $m @300ep"
}
Start-Sleep -Seconds 10
Say "waiting for v65/v66/v68 300ep (up to 7h)..."
WaitScores @('v65_300ep_train','v66_300ep_train','v68_300ep_train') 420 | Out-Null

# ---------- Phase 3: final ensembling ----------
Say "Phase 3: saving TTA preds for v65/v66/v68 + final ensemble sweep..."
& $py "$proj\misc\save_val_tta_preds.py"  2>&1 | Add-Content $log
& $py "$proj\misc\save_test_tta_preds.py" 2>&1 | Add-Content $log
& $py "$proj\misc\overnight_ensemble.py" 2>&1 | Add-Content $log

# collect final single-model scores
Say "=== FINAL SINGLE-MODEL SCORES ==="
foreach($m in @('v60_w5_300ep','v60_w8_300ep','v60_w3_300ep','v65_300ep','v66_300ep','v68_300ep')){
  $f = "$proj\models\logs\${m}_train.log"
  if (Test-Path $f){ $s=(Select-String -Path $f -Pattern "AGENT_SCORE" -EA SilentlyContinue).Line; Say "$m : $s" }
}
Say "=== OVERNIGHT DONE === see OVERNIGHT_RESULTS.md + models/solutions/solution_overnight_*.csv"
