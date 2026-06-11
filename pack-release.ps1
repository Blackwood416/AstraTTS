# AstraTTS 资源与发布整合打包脚本
# 设置控制台输出编码为 UTF8，防止中文乱码
chcp 65001 >$null
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Version = "v1.2.2"
$ReleaseDir = "releases"

if (!(Test-Path $ReleaseDir)) {
    New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
}

$7z_exe = "7z"
if (!(Get-Command $7z_exe -ErrorAction SilentlyContinue)) {
    if (Test-Path "$env:ProgramFiles\7-Zip\7z.exe") {
        $7z_exe = "$env:ProgramFiles\7-Zip\7z.exe"
    } else {
        $7z_exe = $null
    }
}

function Compress-WithProgress {
    param([string]$SourcePath, [string]$DestinationPath)
    if ($7z_exe) {
        Write-Host "  👉 正在使用 7-Zip 进行高速压缩..." -ForegroundColor DarkCyan
        & $7z_exe a -tzip "$DestinationPath" "$SourcePath" -bso0 -bsp1
    } else {
        Write-Host "  ⚠️ 未检测到 7-Zip，正使用系统自带压缩 (较慢且无进度条，请耐心等待)..." -ForegroundColor DarkGray
        Compress-Archive -Path $SourcePath -DestinationPath $DestinationPath -Force
    }
}

Write-Host "📦 开始打包 AstraTTS $Version 发布文件..." -ForegroundColor Cyan

# 打包 Windows 整合包
if (Test-Path "publish") {
    $winZip = "$ReleaseDir/AstraTTS-$Version-win64.zip"
    if (Test-Path $winZip) { Remove-Item $winZip -Force }
    Write-Host "`n正在压缩 Windows 整合包 -> $winZip ..." -ForegroundColor Yellow
    Compress-WithProgress -SourcePath "publish\*" -DestinationPath $winZip
    Write-Host "✅ Windows 整合包打包完成！" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ 未找到 publish 目录，跳过打包 Windows。" -ForegroundColor Yellow
}

# 打包独立资源包 (resources-minimal)
if (Test-Path "resources-minimal") {
    $resZip = "$ReleaseDir/AstraTTS-resources-minimal-$Version.zip"
    if (Test-Path $resZip) { Remove-Item $resZip -Force }
    Write-Host "`n正在压缩独立资源包 (resources-minimal) -> $resZip ..." -ForegroundColor Yellow
    Compress-WithProgress -SourcePath "resources-minimal" -DestinationPath $resZip
    Write-Host "✅ 独立资源包打包完成！" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ 未找到 resources-minimal 目录，跳过打包核心资源。" -ForegroundColor Yellow
}

Write-Host "`n🎉 所有打包作业均已完成进入: $(Resolve-Path $ReleaseDir)" -ForegroundColor Cyan
