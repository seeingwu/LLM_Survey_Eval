# Start backend + frontend from repo root
# Usage: .\start_dev.ps1

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

Write-Host '== Backend =='
$backendPath = Join-Path $repoRoot 'agent\backend'
Set-Location $backendPath

if (-not (Test-Path '.venv')) {
  python -m venv .venv
}

$activate = '.\.venv\Scripts\Activate.ps1'
& $activate

$reqPath = Join-Path $backendPath 'requirements.txt'
$projPath = Join-Path $repoRoot 'pyproject.toml'
$stampPath = Join-Path $backendPath '.venv\.deps_stamp'
$reqHash = (Get-FileHash $reqPath).Hash
$projHash = (Get-FileHash $projPath).Hash
$stampValue = "$reqHash|$projHash"

if (-not (Test-Path $stampPath) -or (Get-Content $stampPath -ErrorAction SilentlyContinue) -ne $stampValue) {
  pip install -e $repoRoot
  pip install -r $reqPath
  $stampValue | Set-Content -Path $stampPath
} else {
  Write-Host 'Backend deps unchanged, skip pip install.'
}

Start-Process -NoNewWindow powershell -ArgumentList @(
  '-NoExit',
  '-Command',
  "Set-Location -LiteralPath '$backendPath'; & `"$activate`"; uvicorn main:app --reload --port 8000"
)

Write-Host '== Frontend =='
$frontendPath = Join-Path $repoRoot 'agent\frontend'
Set-Location $frontendPath

$frontendStamp = Join-Path $frontendPath '.deps_stamp'
$frontendLock = Join-Path $frontendPath 'package-lock.json'
$frontendPkg = Join-Path $frontendPath 'package.json'
$frontendHashTarget = $frontendLock
if (-not (Test-Path $frontendLock)) {
  $frontendHashTarget = $frontendPkg
}
$frontendHash = (Get-FileHash $frontendHashTarget).Hash

if (-not (Test-Path 'node_modules') -or -not (Test-Path $frontendStamp) -or (Get-Content $frontendStamp -ErrorAction SilentlyContinue) -ne $frontendHash) {
  npm install
  $frontendHash | Set-Content -Path $frontendStamp
} else {
  Write-Host 'Frontend deps unchanged, skip npm install.'
}

Start-Process -NoNewWindow powershell -ArgumentList @(
  '-NoExit',
  '-Command',
  "Set-Location -LiteralPath '$frontendPath'; npm run dev"
)

Write-Host 'Started backend on http://127.0.0.1:8000 and frontend on http://localhost:5173'
