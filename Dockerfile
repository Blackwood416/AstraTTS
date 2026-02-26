# ==========================================
# 阶段 1: 编译构建 C# 项目
# ==========================================
ARG REGISTRY=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/mcr.microsoft.com
FROM ${REGISTRY}/dotnet/sdk:10.0 AS build
WORKDIR /src

# 拷贝依赖文件并还原
COPY ["AstraTTS.Web/AstraTTS.Web.csproj", "AstraTTS.Web/"]
COPY ["AstraTTS.Core/AstraTTS.Core.csproj", "AstraTTS.Core/"]
COPY ["AstraTTS.CLI/AstraTTS.CLI.csproj", "AstraTTS.CLI/"]
RUN dotnet restore "AstraTTS.Web/AstraTTS.Web.csproj"

# 拷贝全部源码
COPY . .

# 编译并发布 Web 项目
WORKDIR "/src/AstraTTS.Web"
RUN dotnet publish "AstraTTS.Web.csproj" -c Release -o /app/publish /p:UseAppHost=false

# 准备配置文件和跨平台工具脚本
WORKDIR "/src"
RUN cp config.template.yaml /app/publish/config.template.yaml && \
    cp config.template.yaml /app/publish/config.yaml && \
    # 拷贝 tools/converter 目录(移除无用的 Windows runtime)
    mkdir -p /app/publish/tools/converter && \
    cp -r tools/converter/v1_converter.py /app/publish/tools/converter/ && \
    cp -r tools/converter/templates /app/publish/tools/converter/

# ==========================================
# 阶段 2: 最终运行环境 (包含 .NET 10, Python 3.12, 和 模型资源)
# ==========================================
ARG REGISTRY=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/mcr.microsoft.com
FROM ${REGISTRY}/dotnet/aspnet:10.0 AS final
WORKDIR /app

# 安装依赖 (针对 Ubuntu Noble)
# 1. 系统依赖
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources || true && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python 环境与依赖 (强制安装 CPU 版 torch 以缩减数 GB 的体积)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    onnx numpy onnxsim onnxruntime && \
    pip install --no-cache-dir torch -f https://mirrors.aliyun.com/pytorch-wheels/cpu

# 从构建阶段拷贝编译好的程序
COPY --from=build /app/publish .

# 拷贝 LFS 追踪的资源文件夹 (避免在 build 阶段带入导致双重冗余)
COPY resources-minimal /app/resources

# 暴露端口
EXPOSE 5000
ENV ASPNETCORE_URLS=http://+:5000

# 运行 Web 服务
ENTRYPOINT ["dotnet", "astra-server.dll"]
