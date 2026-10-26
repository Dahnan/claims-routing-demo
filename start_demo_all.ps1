# D:\Documents\Python\System_1_Demo\start_demo_all.ps1
# Master launch script for the System 1 vs System 2 Claims Router Demo.
# Launches:
#   1. llama-server (127.0.0.1:8000) with Gemma 4 12B Instruct (Q8_0)
#   2. FastAPI Web Application & Dashboard (127.0.0.1:8080)
#   3. Microsoft Dev Tunnel (devtunnel host claims-demo) for VPN bypass (if installed)

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host " HEALTH INSURANCE CLAIMS ROUTER: SYSTEM 1 VS SYSTEM 2 DEMO LAUNCHER     " -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# 1. Check prerequisites
$ModelPath = "$PSScriptRoot\models\gemma-4-12B-it-Q8_0.gguf"
if (-not (Test-Path $ModelPath)) {
    Write-Host "[ERROR] Model file not found at: $ModelPath" -ForegroundColor Red
    exit 1
}

$PythonExe = "C:\Users\dahna\miniconda3\envs\System1Env\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
}

Write-Host "[OK] Using Python environment: $PythonExe" -ForegroundColor Green

# 2. Check if llama-server is already running on port 8000
$Port8000Active = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if (-not $Port8000Active) {
    Write-Host "Starting llama-server in background window on 127.0.0.1:8000..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\start_model_server.ps1`""
    
    Write-Host "Waiting for llama-server to report healthy..." -NoNewline
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 2 -ErrorAction Stop
            if ($resp.status -eq "ok") {
                $ready = $true
                break
            }
        } catch {
            Write-Host "." -NoNewline
        }
    }
    Write-Host ""
    if (-not $ready) {
        Write-Host "[WARNING] llama-server took longer than expected to start. Continuing..." -ForegroundColor Yellow
    } else {
        Write-Host "[OK] llama-server is online and healthy!" -ForegroundColor Green
    }
} else {
    Write-Host "[OK] llama-server is already running on 127.0.0.1:8000." -ForegroundColor Green
}

# 3. Check devtunnel
$DevTunnel = Get-Command devtunnel -ErrorAction SilentlyContinue
if ($DevTunnel) {
    Write-Host "[OK] Microsoft Dev Tunnels is installed." -ForegroundColor Green
    Write-Host "To expose port 8080 to your corporate VPN laptop:" -ForegroundColor Cyan
    Write-Host "   devtunnel host claims-demo" -ForegroundColor White
} else {
    Write-Host "[INFO] devtunnel is not on PATH. To install for presentation over VPN:" -ForegroundColor Gray
    Write-Host "   winget install Microsoft.devtunnel" -ForegroundColor Gray
}

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host " Starting FastAPI Dashboard on http://127.0.0.1:8080 ...                " -ForegroundColor Green
Write-Host " Open your browser to: http://127.0.0.1:8080                            " -ForegroundColor White
Write-Host " Press Ctrl+C in this terminal to stop the web server.                 " -ForegroundColor Gray
Write-Host "========================================================================" -ForegroundColor Cyan

& $PythonExe -m uvicorn server:app --host 127.0.0.1 --port 8080 --reload
