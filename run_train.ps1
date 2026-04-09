#!/usr/bin/env pwsh
# Wrapper to run training with venv activation
# This ensures hydra and all dependencies are available

$venvPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $PSScriptRoot "scripts\train_supervised.py"

# Run Python with full venv path and all arguments
& $venvPath $scriptPath @args
exit $LASTEXITCODE
