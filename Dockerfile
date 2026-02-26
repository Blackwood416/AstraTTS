# ==========================================
# AstraTTS 统一单阶段构建镜像 (基于 .NET 10 SDK)
# ==========================================
# 使用华为云镜像加速前缀
ARG REGISTRY=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/mcr.microsoft.com
FROM ${REGISTRY}/dotnet/sdk:10.0
WORKDIR /app

# 1. 基础环境配置 (Ubuntu Noble)
# 替换为清华源并安装 Python 3.12 及必要工具
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources || true && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Python 环境与依赖 (强制 CPU 版 Torch 以缩减体积)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    onnx numpy onnxsim onnxruntime && \
    pip install --no-cache-dir torch -f https://mirrors.aliyun.com/pytorch-wheels/cpu

# 3. 编译 C# 项目
# 拷贝依赖文件并还原
COPY ["AstraTTS.Web/AstraTTS.Web.csproj", "AstraTTS.Web/"]
COPY ["AstraTTS.Core/AstraTTS.Core.csproj", "AstraTTS.Core/"]
COPY ["AstraTTS.CLI/AstraTTS.CLI.csproj", "AstraTTS.CLI/"]
RUN dotnet restore "AstraTTS.Web/AstraTTS.Web.csproj"

# 拷贝全量源码
COPY . .

# 发布项目到 /app/publish
WORKDIR "/app/AstraTTS.Web"
RUN dotnet publish "AstraTTS.Web.csproj" -c Release -o /app/publish /p:UseAppHost=false

# 4. 准备运行态文件
WORKDIR /app/publish
# 拷贝 LFS 模型资源
COPY resources-minimal ./resources
# 生成基础配置
RUN cp /app/config.template.yaml ./config.template.yaml && \
    cp /app/config.template.yaml ./config.yaml && \
    # 拷贝转换器脚本 (保持目录结构)
    mkdir -p ./tools/converter && \
    cp /app/tools/converter/v1_converter.py ./tools/converter/ && \
    cp -r /app/tools/converter/templates ./tools/converter/

# 5. 运行配置
EXPOSE 5000
ENV ASPNETCORE_URLS=http://+:5000
# 镜像启动即进入发布目录
ENTRYPOINT ["dotnet", "astra-server.dll"]
