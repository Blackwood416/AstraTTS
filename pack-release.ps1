# AstraTTS 资源与发布整合打包脚本
# 设置控制台输出编码为 UTF8，防止中文乱码
chcp 65001 >$null
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Version = "v1.1.1"
$ReleaseDir = "releases"

if (!(Test-Path $ReleaseDir)) {
    New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
}

Write-Host "📦 开始打包 AstraTTS $Version 发布文件..." -ForegroundColor Cyan

# 1. 打包 Windows 整合包 (假设已经运行了 publish.ps1)
if (Test-Path "publish") {
    $winZip = "$ReleaseDir/AstraTTS-$Version-win64.zip"
    if (Test-Path $winZip) { Remove-Item $winZip -Force }
    Write-Host "`n正在压缩 Windows 整合包 -> $winZip ..." -ForegroundColor Yellow
    Compress-Archive -Path "publish/*" -DestinationPath $winZip -Force
    Write-Host "✅ Windows 整合包打包完成！" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ 未找到 publish 目录，跳过打包 Windows。提示: 请先运行 .\publish.ps1" -ForegroundColor Yellow
}

# 2. 打包 Linux 整合包 (假设已经运行了 publish-linux.sh)
if (Test-Path "publish-linux") {
    $linuxZip = "$ReleaseDir/AstraTTS-$Version-linux64.zip"
    if (Test-Path $linuxZip) { Remove-Item $linuxZip -Force }
    Write-Host "`n正在压缩 Linux 整合包 -> $linuxZip ..." -ForegroundColor Yellow
    Compress-Archive -Path "publish-linux/*" -DestinationPath $linuxZip -Force
    Write-Host "✅ Linux 整合包打包完成！" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ 未找到 publish-linux 目录，跳过打包 Linux。提示: 请先运行 ./publish-linux.sh" -ForegroundColor Yellow
}

# 3. 单独打包大体积的资源依赖库 (适用于只需要部署 Docker 或已有主程序只需升级资源的用户)
if (Test-Path "resources-minimal") {
    $resZip = "$ReleaseDir/AstraTTS-resources-minimal-$Version.zip"
    if (Test-Path $resZip) { Remove-Item $resZip -Force }
    Write-Host "`n正在压缩独立资源包 (resources-minimal) -> $resZip ..." -ForegroundColor Yellow
    Compress-Archive -Path "resources-minimal" -DestinationPath $resZip -Force
    Write-Host "✅ 独立资源包打包完成！" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ 未找到 resources-minimal 目录，跳过核心资源的单独打包。" -ForegroundColor Red
}

Write-Host "`n🎉 所有打包作业均已进入: $(Resolve-Path $ReleaseDir)" -ForegroundColor Cyan
Write-Host "提示: 现在可以直接前往 Releases 页面上传 '$ReleaseDir' 内的文件了。" -ForegroundColor Gray
