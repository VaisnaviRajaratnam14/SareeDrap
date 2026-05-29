# SareeDrape Studio — Stop All Services
# Run from project root: powershell -ExecutionPolicy Bypass -File stop.ps1

Write-Host "Stopping SareeDrape Studio services..." -ForegroundColor Yellow

# Stop Vite (Node on 5173)
$pid5173 = (netstat -ano | Select-String ":5173.*LISTENING" | ForEach-Object { ($_ -split '\s+')[-1] }) | Select-Object -First 1
if ($pid5173 -match '^\d+$') { Stop-Process -Id ([int]$pid5173) -Force -ErrorAction SilentlyContinue; Write-Host "[Vite]    Stopped" -ForegroundColor Green }

# Stop Flask (Python on 5000)
Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "[Flask]   Stopped" -ForegroundColor Green

# Stop MongoDB (mongod on 27017)
Get-Process -Name mongod -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "[MongoDB] Stopped" -ForegroundColor Green

Write-Host "All services stopped." -ForegroundColor Cyan
