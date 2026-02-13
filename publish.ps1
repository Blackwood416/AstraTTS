# AstraTTS 统一发布脚本

$PublishDir = "publish"
$Runtime = "win-x64" # 可以根据需要修改，例如 win-arm64

if (Test-Path $PublishDir) {
    Write-Host "正在清理旧的发布目录..." -ForegroundColor Cyan
    Remove-Item -Path $PublishDir -Force -Recurse
}

Write-Host "🚀 开始发布 AstraTTS.Web (astra-server)..." -ForegroundColor Green
dotnet publish AstraTTS.Web/AstraTTS.Web.csproj -c Release -r $Runtime --self-contained true -o $PublishDir /p:PublishSingleFile=false

Write-Host "🚀 开始发布 AstraTTS.CLI (astra-cli)..." -ForegroundColor Green
dotnet publish AstraTTS.CLI/AstraTTS.CLI.csproj -c Release -r $Runtime --self-contained true -o $PublishDir /p:PublishSingleFile=false

# 复制配置文件模板
#if (Test-Path "config.template.json") {
    #Write-Host "复制配置文件模板..." -ForegroundColor Yellow
    #Copy-Item "config.template.json" -Destination "$PublishDir/config.template.json"
#}

# 复制模型转换工具
#Write-Host "--- 集成模型转换工具 ---" -ForegroundColor Cyan
#$toolsDir = "$PublishDir/tools/converter"
#if (!(Test-Path "$toolsDir/templates")) {
    #New-Item -ItemType Directory -Force -Path "$toolsDir/templates" | Out-Null
#}
#Copy-Item "AstraTTS.Core/scripts/v1_converter.py" -Destination "$toolsDir/v1_converter.py"
#Copy-Item "AstraTTS.Core/scripts/init_env.ps1" -Destination "$toolsDir/init_env.ps1"
#Copy-Item "AstraTTS.Core/scripts/templates/*.onnx" -Destination "$toolsDir/templates/"

Write-Host "`n✅ 发布完成！所有文件已整合至: $(Resolve-Path $PublishDir)" -ForegroundColor Green
Write-Host "运行提示:"
Write-Host "  - 运行 Web 服务: ./$PublishDir/astra-server.exe"
Write-Host "  - 运行 CLI 工具: ./$PublishDir/astra-cli.exe"
