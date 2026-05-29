# SareeDrape Studio — Start All Services
# Run from project root: powershell -ExecutionPolicy Bypass -File start.ps1

$ROOT    = $PSScriptRoot
$MONGOD  = "d:\mongodb\bin\mongod.exe"
$DBPATH  = "d:\mongodata"
$LOGPATH = "d:\mongolog\mongod.log"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   SareeDrape Studio — Starting Up      " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. MongoDB ──────────────────────────────────────────────────────────────
$mongoRunning = netstat -ano | Select-String ":27017.*LISTENING"
if ($mongoRunning) {
    Write-Host "[MongoDB]  Already running on port 27017" -ForegroundColor Green
} else {
    if (-not (Test-Path $MONGOD)) {
        Write-Host "[MongoDB]  mongod.exe not found at $MONGOD" -ForegroundColor Red
        Write-Host "           Download: https://www.mongodb.com/try/download/community"
        exit 1
    }
    New-Item -ItemType Directory -Path $DBPATH  -Force | Out-Null
    New-Item -ItemType Directory -Path (Split-Path $LOGPATH) -Force | Out-Null
    Start-Process -FilePath $MONGOD -ArgumentList "--dbpath `"$DBPATH`" --logpath `"$LOGPATH`" --port 27017" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Host "[MongoDB]  Started  → port 27017" -ForegroundColor Green
}

# ── 2. Flask backend ────────────────────────────────────────────────────────
$flaskRunning = netstat -ano | Select-String ":5000.*LISTENING"
if ($flaskRunning) {
    Write-Host "[Flask]    Already running on port 5000" -ForegroundColor Green
} else {
    $venv = Join-Path $ROOT "backend\venv\Scripts\python.exe"
    if (-not (Test-Path $venv)) {
        Write-Host "[Flask]    venv not found. Run: cd backend && python -m venv venv && venv\Scripts\pip install -r requirements.txt" -ForegroundColor Red
        exit 1
    }
    Start-Process -FilePath $venv -ArgumentList "app.py" -WorkingDirectory (Join-Path $ROOT "backend") -WindowStyle Normal
    Start-Sleep -Seconds 3
    Write-Host "[Flask]    Started  → http://localhost:5000" -ForegroundColor Green
}

# ── 3. React / Vite frontend ────────────────────────────────────────────────
$viteRunning = netstat -ano | Select-String ":5173.*LISTENING"
if ($viteRunning) {
    Write-Host "[Vite]     Already running on port 5173" -ForegroundColor Green
} else {
    $npm = (Get-Command npm -ErrorAction SilentlyContinue)?.Source
    if (-not $npm) {
        Write-Host "[Vite]     npm not found in PATH" -ForegroundColor Red
        exit 1
    }
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c npm run dev" -WorkingDirectory (Join-Path $ROOT "frontend") -WindowStyle Normal
    Start-Sleep -Seconds 4
    Write-Host "[Vite]     Started  → http://localhost:5173" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   All services running!" -ForegroundColor Green
Write-Host ""
Write-Host "   Frontend  → http://localhost:5173" -ForegroundColor White
Write-Host "   Backend   → http://localhost:5000" -ForegroundColor White
Write-Host "   API docs  → http://localhost:5000/api/health" -ForegroundColor White
Write-Host ""
Write-Host "   Demo login : demo@saree.com  / demo123" -ForegroundColor Yellow
Write-Host "   Admin login: admin@saree.com / admin123" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
