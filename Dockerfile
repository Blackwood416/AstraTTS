# ==========================================
# 阶段 1: 编译构建 C# 项目
# ==========================================
# 允许构建时传入镜像加速前缀，默认为 mcr官方
ARG REGISTRY=mcr.microsoft.com
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
    cp config.template.yaml /app/publish/config.yaml

# 拷贝 tools/converter 目录(移除无用的 Windows runtime，只保留脚本和模板)
RUN mkdir -p /app/publish/tools/converter && \
    cp -r tools/converter/v1_converter.py /app/publish/tools/converter/ && \
    cp -r tools/converter/templates /app/publish/tools/converter/

# ==========================================
# 阶段 2: 最终运行环境 (包含 .NET 10, Python 3.12, 和 模型资源)
# ==========================================
# ubuntu noble 自带 Python 3.12, onnxsim 在 Linux 上支持python 3.12
ARG REGISTRY=mcr.microsoft.com
FROM ${REGISTRY}/dotnet/aspnet:10.0 AS final
WORKDIR /app

# 替换默认源为清华源以加速国内 apt 下载 (针对 Ubuntu Noble)
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources || true

# 安装 Python 3.11 和 wget, unzip (用于下载资源)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 创建并激活 python 虚拟环境以安装依赖 (避免打破系统包管理)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 安装转换器依赖 (使用清华 pip 源加速)
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple onnx torch numpy onnxsim onnxruntime

# 使用通配符尝试拷贝本地可能存在的 resources-minimal
# 附带 config.template.yaml 作为兜底文件防止本地没有资源时 COPY 失败报错
COPY ["config.template.yaml", "resources-minima[l]", "/app/local_res/"]

# 检查本地资源，如果不存在则从 R2 云端下载
RUN if [ -d "/app/local_res/models_v1" ]; then \
        echo "📦 检测到本地包含 resources-minimal，直接使用本地资源..." && \
        mkdir -p /app/resources && \
        mv /app/local_res/models_v1 /app/resources/ && \
        mv /app/local_res/avatars /app/resources/ 2>/dev/null || true; \
        mv /app/local_res/shared /app/resources/ 2>/dev/null || true; \
    else \
        echo "☁️ 本地未包含有效资源，正在从 R2 下载..." && \
        wget -q "https://r2.blackwood.cv/share/73b8b7e12b" -O resources-minimal.zip && \
        unzip -q resources-minimal.zip && \
        mv resources-minimal resources && \
        rm resources-minimal.zip; \
    fi && \
    rm -rf /app/local_res

# 从构建阶段拷贝编译好的程序
COPY --from=build /app/publish .

# 暴露端口
EXPOSE 5000
ENV ASPNETCORE_URLS=http://+:5000

# 运行 Web 服务
ENTRYPOINT ["dotnet", "AstraTTS.Web.dll"]
