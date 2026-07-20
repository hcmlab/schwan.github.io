# 06_check_corrupt_windows.ps1
# PowerShell script to verify video integrity on Windows

# 1. Set Environment Variables
$env:DATA_ROOT = "X:\data\Schwan_T3_FineTune"
$env:OUT_DIR = "gpu_server/data"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Verifying Video Integrity on Windows" -ForegroundColor Cyan
Write-Host "DATA_ROOT: $env:DATA_ROOT"
Write-Host "OUT_DIR:   $env:OUT_DIR"
Write-Host "============================================================" -ForegroundColor Cyan

# 2. Check if DATA_ROOT exists
if (-not (Test-Path $env:DATA_ROOT)) {
    Write-Host "Error: DATA_ROOT $env:DATA_ROOT not found." -ForegroundColor Red
    exit 1
}

# 3. Activate Conda Environment
# Note: This assumes conda is in the PATH. 
# We use 'conda run' to avoid issues with PowerShell activation in a script.
$condaEnv = "vlm_annotation"

Write-Host "Using conda environment: $condaEnv"
Write-Host "Running corruption check (Dry Run)..."

# 4. Run the Python script
# We run from the repo root
conda run -n $condaEnv python gpu_server/src/quarantine_corrupt_videos.py --dry-run

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Check complete. Review the output above for corrupted files." -ForegroundColor Cyan
Write-Host "To actually quarantine these files, run the script without --dry-run." -ForegroundColor Cyan
