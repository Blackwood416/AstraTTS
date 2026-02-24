# AstraTTS 统一发布脚本
# 设置控制台输出编码
chcp 65001 >$null
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 测试输出
Write-Host "--- AstraTTS 编译发布工具 ---" -ForegroundColor Cyan

$PublishDir = "publish"
$Runtime = "win-x64" # 可以根据需要修改，例如 win-arm64

if (Test-Path $PublishDir) {
    Write-Host "正在清理旧的发布目录..." -ForegroundColor Cyan
    Remove-Item -Path $PublishDir -Force -Recurse
}

Write-Host "🚀 开始发布 AstraTTS.Web (astra-server)..." -ForegroundColor Green
dotnet publish AstraTTS.Web/AstraTTS.Web.csproj -c Release -r $Runtime --self-contained true -o $PublishDir /p:PublishSingleFile=false /p:AllowMissingPrunePackageData=true

Write-Host "🚀 开始发布 AstraTTS.CLI (astra-cli)..." -ForegroundColor Green
dotnet publish AstraTTS.CLI/AstraTTS.CLI.csproj -c Release -r $Runtime --self-contained true -o $PublishDir /p:PublishSingleFile=false /p:AllowMissingPrunePackageData=true

# 清理旧的 config.json (如果残留)
if (Test-Path "$PublishDir/config.json") {
    Remove-Item "$PublishDir/config.json" -Force
}

# 复制配置文件模板
if (Test-Path "config.template.yaml") {
    Write-Host "📦 正在复制配置文件..." -ForegroundColor Yellow
    Copy-Item "config.template.yaml" -Destination "$PublishDir/config.template.yaml"
    Copy-Item "config.template.yaml" -Destination "$PublishDir/config.yaml"
}

# 复制最小化资源目录
if (Test-Path "resources-minimal") {
    Write-Host "📦 正在复制资源文件 (minimal)..." -ForegroundColor Yellow
    Copy-Item -Path "resources-minimal" -Destination "$PublishDir/resources" -Recurse -Force
}

# 复制工具目录
if (Test-Path "tools") {
    Write-Host "📦 正在复制工具目录..." -ForegroundColor Yellow
    Copy-Item -Path "tools" -Destination "$PublishDir/tools" -Recurse -Force
}

Write-Host "`n✅ 发布完成！所有文件已整合至: $(Resolve-Path $PublishDir)" -ForegroundColor Green
Write-Host "运行提示:"
Write-Host "  - 运行 Web 服务: ./$PublishDir/astra-server.exe"
Write-Host "  - 运行 CLI 工具: ./$PublishDir/astra-cli.exe"
