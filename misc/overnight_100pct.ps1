# Queue: retrain the ensemble's better members on 100% data (train+val), 4 at a time, AFTER the
# current GPU work (v72/v72_w8/v73) frees up. Saves model_<tag>_full.pth for each. The final
# ensemble re-apply (regenerate _full test preds + reapply the 80%-fit 2-group weights) is a quick
# follow-up the lead runs once these finish.
$ErrorActionPreference = 'Continue'
$proj = "C:\Users\jan\Downloads\nml_project_proposal\nml_project_proposal"
$py   = "C:\Users\jan\Downloads\nml_project_proposal\.venv\Scripts\python.exe"
$log  = "$proj\models\logs\retrain100.log"
function Say($m){ Add-Content $log "[$(Get-Date -Format HH:mm:ss)] $m"; Write-Output "[$(Get-Date -Format HH:mm:ss)] $m" }

# tag -> model .py (relative to proj). 'better models' = base + ball pool, non-SWA (SWA needs
# snapshot averaging; keep 80% SWA in the ensemble). 14 models.
$jobs = @(
  @('v46','models/nba_v46.py'), @('v58','models/nba_v58.py'),
  @('v46_s3','models/nba_v46_s3.py'), @('v46_s5','models/nba_v46_s5.py'), @('v46_s7','models/nba_v46_s7.py'),
  @('v58_s1','models/nba_v58_s1.py'), @('v58_s2','models/nba_v58_s2.py'), @('v58_s3','models/nba_v58_s3.py'),
  @('v60_w5','models/nba_v60_w5.py'), @('v60_w8','models/nba_v60_w8.py'), @('v60_w3','models/nba_v60_w3.py'),
  @('v60_w8_s1','models/nba_v60_w8_s1.py'), @('v60_w8_s2','models/nba_v60_w8_s2.py'), @('v60_w5_s1','models/nba_v60_w5_s1.py')
)

Say "=== 100% RETRAIN QUEUE START: waiting for current GPU jobs (v72/v72_w8/v73) ==="
$deadline = (Get-Date).AddHours(3)
while ((Get-Date) -lt $deadline) {
  $busy = (Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match 'run_one' }).Count
  if ($busy -eq 0) { break }
  Start-Sleep -Seconds 60
}
Say "GPU free. Retraining $($jobs.Count) models on 100% data, 4 at a time."

$batch = 4
for ($i = 0; $i -lt $jobs.Count; $i += $batch) {
  $slice = $jobs[$i..([Math]::Min($i + $batch - 1, $jobs.Count - 1))]
  foreach ($j in $slice) {
    $tag = $j[0]; $mp = $j[1]
    Start-Process -WindowStyle Hidden -FilePath $py `
      -ArgumentList "$proj\misc\retrain_full.py $mp $tag 300" `
      -RedirectStandardOutput "$proj\models\logs\retrain100_$tag.log" `
      -RedirectStandardError  "$proj\models\logs\retrain100_${tag}_err.log" `
      -WorkingDirectory $proj
    Say "launched retrain_full $tag"
  }
  # wait for this batch's weight files
  foreach ($j in $slice) {
    $tag = $j[0]; $wf = "$proj\models\weights\model_${tag}_full.pth"
    $bd = (Get-Date).AddHours(8)
    while (-not (Test-Path $wf) -and (Get-Date) -lt $bd) { Start-Sleep -Seconds 60 }
    if (Test-Path $wf) { Say "done $tag" } else { Say "TIMEOUT/Failed $tag" }
  }
}
Say "=== 100PCT RETRAIN DONE ($($jobs.Count) models). Next: regenerate _full test preds + reapply 2-group weights. ==="
