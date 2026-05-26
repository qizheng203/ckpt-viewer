param(
    [string]$Python = $env:CKPT_VIEWER_PYTHON,
    [string]$Venv = $env:CKPT_VIEWER_VENV,
    [string]$BinDir = $env:CKPT_VIEWER_BIN_DIR
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "[ckpt-viewer] $Message" -ForegroundColor Cyan
}

function Fail([string]$Message) {
    Write-Host "[ckpt-viewer] $Message" -ForegroundColor Red
    exit 1
}

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Venv) {
    $Venv = Join-Path $ProjectDir ".venv"
}
if (-not $BinDir) {
    $BinDir = Join-Path $env:USERPROFILE ".local\bin"
}
$UseSystemSite = $true
if ($env:CKPT_VIEWER_VENV_SYSTEM_SITE -eq "0") {
    $UseSystemSite = $false
}

function Resolve-Python {
    if ($Python) {
        $cmd = Get-Command $Python -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        if (Test-Path $Python) { return (Resolve-Path $Python).Path }
        Fail "CKPT_VIEWER_PYTHON 指向的 Python 不存在：$Python"
    }

    $candidates = @("py", "python", "python3")
    foreach ($candidate in $candidates) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    Fail "未找到 Python。请安装 Python >= 3.10，或设置 CKPT_VIEWER_PYTHON 后重试。"
}

$PythonExe = Resolve-Python
Write-Step "项目目录：$ProjectDir"
Write-Step "Python：$PythonExe"

if ((Split-Path -Leaf $PythonExe).ToLower() -eq "py.exe" -or (Split-Path -Leaf $PythonExe).ToLower() -eq "py") {
    $PythonArgs = @("-3")
} else {
    $PythonArgs = @()
}

function Invoke-SelectedPython([string[]]$Arguments) {
    & $PythonExe @($PythonArgs + $Arguments)
}

Invoke-SelectedPython @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)")
if ($LASTEXITCODE -ne 0) {
    Fail "Python 版本过低。Ckpt_viewer 需要 Python >= 3.10。"
}

$VenvPython = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Step "创建虚拟环境：$Venv"
    if ($UseSystemSite) {
        Invoke-SelectedPython @("-m", "venv", "--system-site-packages", $Venv)
    } else {
        Invoke-SelectedPython @("-m", "venv", $Venv)
    }
}

Write-Step "升级 pip/setuptools/wheel"
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Step "安装 Ckpt_viewer 运行依赖"
& $VenvPython -m pip install -e $ProjectDir

Write-Step "写入用户级启动器：$BinDir"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

$CmdLauncher = @"
@echo off
"$VenvPython" -m ckpt_viewer.cli %*
"@
Set-Content -Path (Join-Path $BinDir "ckpt-viewer.cmd") -Value $CmdLauncher -Encoding ASCII
Set-Content -Path (Join-Path $BinDir "pth-inspector.cmd") -Value $CmdLauncher -Encoding ASCII

$PsLauncher = @"
& "$VenvPython" -m ckpt_viewer.cli @args
"@
Set-Content -Path (Join-Path $BinDir "ckpt-viewer.ps1") -Value $PsLauncher -Encoding UTF8
Set-Content -Path (Join-Path $BinDir "pth-inspector.ps1") -Value $PsLauncher -Encoding UTF8

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$PathParts = @()
if ($UserPath) {
    $PathParts = $UserPath -split ';' | Where-Object { $_ }
}
$AlreadyInPath = $false
foreach ($Part in $PathParts) {
    if ($Part.TrimEnd('\') -ieq $BinDir.TrimEnd('\')) {
        $AlreadyInPath = $true
        break
    }
}

if (-not $AlreadyInPath) {
    Write-Step "将 $BinDir 加入用户 PATH"
    $NewPath = if ($UserPath) { "$UserPath;$BinDir" } else { $BinDir }
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
    $env:Path = "$env:Path;$BinDir"
}

Write-Step "验证安装"
& (Join-Path $BinDir "ckpt-viewer.cmd") --version

Write-Host ""
Write-Host "安装完成。" -ForegroundColor Green
Write-Host ""
Write-Host "当前 PowerShell 可直接运行："
Write-Host "  ckpt-viewer"
Write-Host "  ckpt-viewer analyze C:\path\to\model.pth --out report"
Write-Host ""
Write-Host "新打开任意终端后也可以直接运行："
Write-Host "  ckpt-viewer"
Write-Host ""
Write-Host "如果 Windows PowerShell 阻止脚本执行，请使用："
Write-Host "  powershell -ExecutionPolicy Bypass -File .\install.ps1"
