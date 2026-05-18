# Wait for all nml training jobs to finish, then run SWA -> val preds -> test preds -> weighted ensemble.
# Detached so it survives terminal close. Logs everything to one file.

$proj = "C:\Users\jan\Downloads\nml_project_proposal\nml_project_proposal"
$py = "C:\Users\jan\Downloads\nml_project_proposal\.venv\Scripts\python.exe"
$log = "$proj\models\logs\orchestrate.log"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Add-Content -Path $log -Value $line
}

Log "==== orchestrate.ps1 starting ===="
Log "Waiting for all nba_v*_s* python training processes to finish..."

# Poll every 60s until no training jobs remain.
while ($true) {
    $jobs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'nba_v\d+(_s\d+)?\.py' -and $_.CommandLine -match 'run_one' })
    if ($jobs.Count -eq 0) { break }
    $names = ($jobs | ForEach-Object {
        if ($_.CommandLine -match 'nba_v\d+_s\d+') { $matches[0] } else { 'nba_vXX' }
    }) -join ','
    Log "  $($jobs.Count) jobs still running: $names"
    Start-Sleep -Seconds 60
}
Log "All training jobs finished."

function RunStep($script) {
    Log "---- Running $script ----"
    $out = & $py "$proj\misc\$script" 2>&1
    Add-Content -Path $log -Value $out
    Log "---- Done $script ----"
}

RunStep "make_all_swa.py"
RunStep "save_val_tta_preds.py"
RunStep "save_test_tta_preds.py"
RunStep "weighted_ensemble.py"

Log "==== orchestrate.ps1 complete ===="
