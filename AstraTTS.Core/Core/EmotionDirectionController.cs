using System.Text.Json;

namespace AstraTTS.Core.Core
{
    /// <summary>
    /// 情感方向向量（τ）控制器。
    ///
    /// 原理：情感语音数据集的说话人嵌入均值差构成"情感方向"，
    /// 在推理时将 x_new = x_base + α·||x_base||·τ̂ 注入说话人嵌入（sv_emb, 20480 维），
    /// 通过 prompt_encoder 重算 ge/ge_advanced，引导声码器生成带情感的语音，
    /// 而无需参考情感音频。
    ///
    /// τ 文件由 tools/emotion/extract_tau.py 从 CSEMOTIONS 数据集提取，
    /// 使用"同一说话人内配对差"公式，避免混入说话人身份差异：
    ///     τ_e = mean_{s,i} ( x_e^{s,i} − x_n^{s} )
    /// </summary>
    public sealed class EmotionDirectionController
    {
        /// <summary>中性情感名（不注入）。</summary>
        public const string NeutralEmotion = "neutral";

        private readonly Dictionary<string, float[]> _tau = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>τ 文件所在目录。</summary>
        public string DirectoryPath { get; }

        /// <summary>τ 向量维度（期望与 speaker_encoder 输出一致，20480）。</summary>
        public int Dim { get; private set; }

        /// <summary>是否成功加载了至少一个 τ。</summary>
        public bool Loaded => _tau.Count > 0;

        /// <summary>已加载的情感名集合。</summary>
        public IReadOnlyCollection<string> Emotions => _tau.Keys;

        public EmotionDirectionController(string? dir)
        {
            DirectoryPath = dir ?? string.Empty;
            Reload();
        }

        /// <summary>
        /// 从目录加载 manifest.json 与 tau_*.f32。文件缺失/损坏时静默降级（Loaded=false）。
        /// 所有 τ 在加载时重新归一化，确保注入公式中的 τ̂ 为单位向量。
        /// </summary>
        public void Reload()
        {
            _tau.Clear();
            Dim = 0;
            try
            {
                if (string.IsNullOrEmpty(DirectoryPath) || !Directory.Exists(DirectoryPath)) return;

                var manifestPath = Path.Combine(DirectoryPath, "manifest.json");
                bool readAny = false;
                if (File.Exists(manifestPath))
                {
                    using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
                    var root = doc.RootElement;
                    if (root.TryGetProperty("dim", out var dimEl) && dimEl.ValueKind == JsonValueKind.Number)
                        Dim = dimEl.GetInt32();

                    if (root.TryGetProperty("emotions", out var emos) && emos.ValueKind == JsonValueKind.Object)
                    {
                        foreach (var prop in emos.EnumerateObject())
                        {
                            var file = prop.Value.TryGetProperty("file", out var f) ? f.GetString() : null;
                            if (string.IsNullOrEmpty(file)) file = $"tau_{prop.Name}.f32";
                            if (TryLoadVector(Path.Combine(DirectoryPath, file), out var vec))
                            {
                                _tau[prop.Name] = vec;
                                readAny = true;
                            }
                        }
                    }
                }

                // manifest 缺失/不完整时，按约定文件名兜底扫描
                if (!readAny)
                {
                    foreach (var file in Directory.EnumerateFiles(DirectoryPath, "tau_*.f32"))
                    {
                        var name = Path.GetFileNameWithoutExtension(file);
                        if (!name.StartsWith("tau_", StringComparison.OrdinalIgnoreCase)) continue;
                        name = name.Substring(4);
                        if (TryLoadVector(file, out var vec))
                        {
                            _tau[name] = vec;
                            readAny = true;
                            if (Dim == 0) Dim = vec.Length;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Emotion] 加载 τ 失败: {ex.Message}");
                _tau.Clear();
            }
        }

        private static bool TryLoadVector(string path, out float[] vec)
        {
            vec = Array.Empty<float>();
            try
            {
                if (!File.Exists(path)) return false;
                var bytes = File.ReadAllBytes(path);
                if (bytes.Length < 4 || bytes.Length % 4 != 0) return false;
                var arr = new float[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
                Normalize(arr);
                vec = arr;
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static void Normalize(float[] v)
        {
            double sum = 0;
            for (int i = 0; i < v.Length; i++) sum += (double)v[i] * v[i];
            double norm = Math.Sqrt(sum);
            if (norm < 1e-12) return;
            float inv = (float)(1.0 / norm);
            for (int i = 0; i < v.Length; i++) v[i] *= inv;
        }

        /// <summary>获取指定情感的归一化 τ（不存在返回 null）。</summary>
        public float[]? GetTau(string emotion)
            => _tau.TryGetValue(emotion, out var t) ? t : null;

        /// <summary>
        /// 注入公式：x_new = x_base + α·||x_base||·τ̂。
        /// τ̂ 为单位向量（GetTau 已保证）。
        /// </summary>
        public static float[] Apply(float[] xBase, float[] tauNormalized, float alpha)
        {
            var result = new float[xBase.Length];
            double sum = 0;
            for (int i = 0; i < xBase.Length; i++) sum += (double)xBase[i] * xBase[i];
            float scale = alpha * (float)Math.Sqrt(sum);

            int n = Math.Min(xBase.Length, tauNormalized.Length);
            for (int i = 0; i < n; i++)
                result[i] = xBase[i] + scale * tauNormalized[i];
            if (n < xBase.Length)
                Array.Copy(xBase, n, result, n, xBase.Length - n);
            return result;
        }
    }
}
