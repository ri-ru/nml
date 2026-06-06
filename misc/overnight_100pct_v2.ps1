# Continuation of the 100% retrain queue, 2-at-a-time (per user request). Waits for the in-flight
# batch-1 retrains to finish, then retrains the remaining 10 pool members on 100% data, 2 concurrent.
$ErrorActionPreference = 'Continue'
$proj = "C:\Users\jan\Downloads\nml_project_proposal\nml_project_proposal"
$py   = "C:\Users\jan\Downloads\nml_project_proposal\.venv\Scripts\python.exe"
$log  = "$proj\models\logs\retrain100.log"
function Say($m){ Add-Content $log "[$(Get-Date -Format HH:mm:ss)] $m"; Write-Output "[$(Get-Date -Format HH:mm:ss)] $m" }

$jobs = @(
  @('v46_s7','models/nba_v46_s7.py'),
  @('v58_s1','models/nba_v58_s1.py'), @('v58_s2','models/nba_v58_s2.py'), @('v58_s3','models/nba_v58_s3.py'),
  @('v60_w5','models/nba_v60_w5.py'), @('v60_w8','models/nba_v60_w8.py'), @('v60_w3','models/nba_v60_w3.py'),
  @('v60_w8_s1','models/nba_v60_w8_s1.py'), @('v60_w8_s2','models/nba_v60_w8_s2.py'), @('v60_w5_s1','models/nba_v60_w5_s1.py')
)

Say "=== v2 QUEUE (2-at-a-time): waiting for batch-1 retrains to finish ==="
$deadline = (Get-Date).AddHours(4)
while ((Get-Date) -lt $deadline) {
  $busy = (Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match 'retrain_full' }).Count
  if ($busy -eq 0) { break }
  Start-Sleep -Seconds 60
}
Say "batch-1 done. Retraining remaining $($jobs.Count) models 2-at-a-time."

$batch = 2
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
  foreach ($j in $slice) {
    $tag = $j[0]; $wf = "$proj\models\weights\model_${tag}_full.pth"
    $bd = (Get-Date).AddHours(8)
    while (-not (Test-Path $wf) -and (Get-Date) -lt $bd) { Start-Sleep -Seconds 60 }
    if (Test-Path $wf) { Say "done $tag" } else { Say "TIMEOUT $tag" }
  }
}
Say "=== 100PCT RETRAIN DONE (all 14). Next: regenerate _full test preds + reapply 2-group weights. ==="
