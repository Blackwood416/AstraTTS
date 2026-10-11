# third_party

## onnxruntime_ggml.dll

[onnxruntime-ggml](https://github.com/thewh1teagle/onnxruntime-ggml) v0.2.0 的
ggml 插件执行提供器（Windows x64）。声码器混合模式（`UseVocoderGgml: true`）需要它，
缺失时该模式自动回退 CPU，不影响其余功能。

获取方式（二选一）：

1. 从 GitHub Releases 下载并解压到本目录：
   ```
   https://github.com/thewh1teagle/onnxruntime-ggml/releases/download/v0.2.0/onnxruntime-ggml-0.2.0-windows-amd64.tar.gz
   ```
   得到 `onnxruntime_ggml.dll` 后放在本目录（构建时自动复制到输出目录）。

2. 不使用构建集成，运行时用环境变量指向 DLL：
   ```
   set ASTRA_GGML_LIBRARY=D:\path\to\onnxruntime_ggml.dll
   ```

其他平台（Linux/macOS）请从同一 Releases 页面下载对应平台包，
文件名分别为 `libonnxruntime_ggml.so` / `libonnxruntime_ggml.dylib`。
