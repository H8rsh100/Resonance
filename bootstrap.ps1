# Windows PowerShell bootstrap for the Citizen Fraud Shield demo.
# PowerShell 5.1 does not support `&&`, so each step is guarded with `if ($?)`.
# Usage:  powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "1/4  Building the labelled corpus"
python -m src.data.build_dataset
if (-not $?) { throw "build_dataset failed" }

Step "2/4  Downloading the public cross-domain corpus"
python -m src.data.fetch_public
if (-not $?) { throw "fetch_public failed (internet required, or run offline and skip)" }

Step "3/4  Training and evaluating"
python -m src.evaluate
if (-not $?) { throw "evaluate failed" }

Step "4/4  Launching the Citizen Fraud Shield"
Write-Host "Opening http://localhost:8501 - press Ctrl+C to stop." -ForegroundColor Green
streamlit run app.py