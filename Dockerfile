# 使用 .NET 10.0 SDK 作为构建阶段
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# 拷贝各项目的依赖文件并还原
COPY ["AstraTTS.Web/AstraTTS.Web.csproj", "AstraTTS.Web/"]
COPY ["AstraTTS.Core/AstraTTS.Core.csproj", "AstraTTS.Core/"]
COPY ["AstraTTS.CLI/AstraTTS.CLI.csproj", "AstraTTS.CLI/"]
RUN dotnet restore "AstraTTS.Web/AstraTTS.Web.csproj"

# 拷贝全部源码
COPY . .

# 编译并发布 Web 项目
WORKDIR "/src/AstraTTS.Web"
RUN dotnet publish "AstraTTS.Web.csproj" -c Release -o /app/publish /p:UseAppHost=false

# 准备资源 (类似 publish-linux.sh 的逻辑)
WORKDIR "/src"
RUN cp config.template.yaml /app/publish/config.template.yaml && \
    cp config.template.yaml /app/publish/config.yaml
RUN if [ -d "resources-minimal" ]; then cp -r resources-minimal /app/publish/resources; fi
RUN if [ -d "tools" ]; then cp -r tools /app/publish/tools; fi

# =============== 运行阶段 ===============
# 使用 ASP.NET Core 运行时作为基础镜像
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
WORKDIR /app

# 安装可能需要的依赖 (如果 ONNX Runtime 或其它库需要)
# RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 从构建阶段拷贝输出
COPY --from=build /app/publish .

# 暴露端口，AstraTTS.Web 默认运行在 5000 或者通过环境变量配置
EXPOSE 5000
ENV ASPNETCORE_URLS=http://+:5000

# 启动 Web 服务器
ENTRYPOINT ["dotnet", "AstraTTS.Web.dll"]
