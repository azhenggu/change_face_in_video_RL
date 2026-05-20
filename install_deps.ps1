# Windows 一键安装脚本
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "安装 Python 依赖..."
pip install -r requirements.txt

Write-Host "安装 InsightFace 预编译包 (Python 3.13)..."
pip install "https://huggingface.co/ussoewwin/Insightface_for_windows/resolve/main/insightface-0.7.3-cp313-cp313-win_amd64.whl"

Write-Host "克隆 vendor 仓库 (GFPGAN / CodeFormer / BasicSR)..."
if (-not (Test-Path vendor)) { New-Item -ItemType Directory vendor | Out-Null }
if (-not (Test-Path vendor/BasicSR)) {
    git clone --depth 1 --branch v1.4.2 https://github.com/XPixelGroup/BasicSR.git vendor/BasicSR
}
if (-not (Test-Path vendor/GFPGAN)) {
    git clone --depth 1 https://github.com/TencentARC/GFPGAN.git vendor/GFPGAN
}
if (-not (Test-Path vendor/CodeFormer)) {
    git clone --depth 1 https://github.com/sczhou/CodeFormer.git vendor/CodeFormer
}

# BasicSR 补丁
$verFile = "vendor/BasicSR/basicsr/version.py"
if (-not (Test-Path $verFile)) {
    @"
__version__ = '1.4.2'
__gitsha__ = 'vendor'
"@ | Set-Content $verFile -Encoding utf8
}

$degFile = "vendor/BasicSR/basicsr/data/degradations.py"
if (Test-Path $degFile) {
    (Get-Content $degFile -Raw) -replace 'from torchvision.transforms.functional_tensor import rgb_to_grayscale', @'
try:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
except ImportError:
    from torchvision.transforms.functional import rgb_to_grayscale
'@ | Set-Content $degFile -Encoding utf8
}

Write-Host "完成。运行: python main.py"
