using Microsoft.ML.OnnxRuntime;
using System.Runtime.InteropServices;

namespace AstraTTS.Core.Core
{
    /// <summary>
    /// onnxruntime-ggml 插件执行提供器的 C# 封装。
    ///
    /// 该提供器把 ONNX 图编译到 ggml 内核（Windows/Linux 上使用 Vulkan，macOS 用 Metal），
    /// 对大规模并行卷积类模型（如声码器 vits）可带来数倍加速。
    /// 已实测：vits 声码器在 AMD RX 5500 XT 上比 CPU 快约 5 倍，频谱包络相关性 0.994。
    ///
    /// 原理：ggml 以插件 EP 形式注册到 ORT（需 ORT 1.29+ 的插件 API），
    /// 注册后作为名为 "ggml" 的 OrtEpDevice 出现，通过
    /// SessionOptions.AppendExecutionProvider(env, devices, options) 挂载到会话。
    ///
    /// 注意：该提供器的图认领策略是"要么全认领、要么不认领"。
    /// 含不支持算子（如 ArgMax）的图会被整图拒绝并静默回落到 CPU EP。
    /// </summary>
    public static class GgmlExecutionProvider
    {
        private static readonly object _lock = new();
        private static bool _registered;
        private static string? _libraryPath;

        /// <summary>
        /// 是否已成功注册（或尝试注册）。
        /// </summary>
        public static bool IsRegistered
        {
            get { lock (_lock) return _registered; }
        }

        /// <summary>
        /// 注册 ggml EP 库（进程内只需一次）。库不存在或注册失败时返回 false，调用方回退 CPU。
        /// </summary>
        public static bool EnsureRegistered(string? libraryPath = null)
        {
            lock (_lock)
            {
                if (_registered) return true;

                try
                {
                    string path = libraryPath ?? ResolveDefaultLibraryPath();
                    if (!File.Exists(path))
                    {
                        return false;
                    }

                    OrtEnv.Instance().RegisterExecutionProviderLibrary("ggml", path);
                    _libraryPath = path;
                    _registered = true;
                    return true;
                }
                catch
                {
                    // 注册失败（缺少 Vulkan 运行库、ORT 版本不匹配等）——静默回退，由调用方走 CPU。
                    return false;
                }
            }
        }

        /// <summary>
        /// 获取 ggml 的 EP 设备列表（未注册或不可用时返回空列表）。
        /// </summary>
        public static IReadOnlyList<OrtEpDevice> GetDevices()
        {
            if (!_registered) return System.Array.Empty<OrtEpDevice>();
            try
            {
                var devices = new List<OrtEpDevice>();
                foreach (var d in OrtEnv.Instance().GetEpDevices())
                {
                    if (string.Equals(d.EpName, "ggml", System.StringComparison.OrdinalIgnoreCase))
                        devices.Add(d);
                }
                return devices;
            }
            catch
            {
                return System.Array.Empty<OrtEpDevice>();
            }
        }

        /// <summary>
        /// 构造挂载 ggml EP 的 SessionOptions；不可用时返回 null（调用方回退 CPU）。
        /// </summary>
        /// <param name="device">gpu / cpu / auto。gpu 需要存在 Vulkan 设备。</param>
        /// <param name="gpuMinWeightMb">
        /// 权重小于该阈值的图强制走 ggml 的 CPU 内核（小图在 GPU 上受启动开销拖累）。
        /// 声码器权重约 250MB，设 0 强制走 GPU。
        /// </param>
        public static SessionOptions? TryCreateSessionOptions(string device = "gpu", int gpuMinWeightMb = 0, bool partial = false, bool profile = false)
        {
            if (!EnsureRegistered()) return null;

            var devices = GetDevices();
            if (devices.Count == 0) return null;

            try
            {
                var so = new SessionOptions();
                var options = new Dictionary<string, string>
                {
                    ["device"] = device,
                    ["gpu_min_weight_mb"] = gpuMinWeightMb.ToString(),
                };
                if (partial) options["partial"] = "1";
                if (profile) options["profile"] = "1";

                so.AppendExecutionProvider(OrtEnv.Instance(), devices, options);
                so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;
                so.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                return so;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// ggml 运行库的默认查找路径：
        ///   1. 环境变量 ASTRA_GGML_LIBRARY（显式指定）
        ///   2. 程序集同目录的 onnxruntime_ggml.dll（随发布包分发）
        /// </summary>
        private static string ResolveDefaultLibraryPath()
        {
            string? env = Environment.GetEnvironmentVariable("ASTRA_GGML_LIBRARY");
            if (!string.IsNullOrWhiteSpace(env)) return env;

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string name = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "onnxruntime_ggml.dll"
                : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                    ? "libonnxruntime_ggml.dylib"
                    : "libonnxruntime_ggml.so";
            return Path.Combine(baseDir, name);
        }
    }
}
