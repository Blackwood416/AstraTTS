using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace AstraTTS.Core.Core
{
    /// <summary>
    /// TTS 推理引擎接口，定义了标准化的推理流程。
    /// </summary>
    public interface ITtsEngine : IDisposable
    {
        /// <summary>
        /// 引擎输出采样率 (Hz)
        /// </summary>
        int SamplingRate { get; }

        /// <summary>
        /// 异步加载引擎所需的模型和资源。
        /// </summary>
        /// <param name="config">TTS 全局配置对象。</param>
        Task LoadAsync(Config.TTSConfig config);

        /// <summary>
        /// 执行异步推理并返回完整的音频数据及元数据。
        /// </summary>
        /// <param name="text">待生成的文本。</param>
        /// <param name="options">推理选项（如语速、噪声比例等）。</param>
        /// <returns>包含音频样本数组和元数据的 TtsResult。</returns>
        Task<TtsResult> PredictAsync(string text, TtsOptions options);

        /// <summary>
        /// 执行流式推理，持续返回音频片段。
        /// </summary>
        /// <param name="text">待生成的文本。</param>
        /// <param name="options">推理选项。</param>
        /// <param name="cancellationToken">取消令牌。</param>
        /// <returns>音频片段的异步流。</returns>
        IAsyncEnumerable<float[]> PredictStreamAsync(string text, TtsOptions options, CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// TTS 推理结果。
    /// </summary>
    public class TtsResult
    {
        public float[] Audio { get; set; } = Array.Empty<float>();
        public int TokenCount { get; set; }
    }

    /// <summary>
    /// 推理过程中的可配置选项。
    /// </summary>
    public class TtsOptions
    {
        /// <summary>
        /// 语速控制 (0.5 - 2.0)。
        /// </summary>
        public float Speed { get; set; } = 1.0f;

        /// <summary>
        /// 噪声系数，影响音色的表现力。
        /// </summary>
        public float NoiseScale { get; set; } = 0.35f;

        /// <summary>
        /// 采样温度，影响生成的多样性。
        /// </summary>
        public float Temperature { get; set; } = 1.0f;

        /// <summary>
        /// Top-K 采样参数。
        /// </summary>
        public int TopK { get; set; } = 15;

        /// <summary>
        /// 流式输出的 Chunk 大小。对于 V1 引擎代表 T2S 采样间隔，对于 V2 引擎代表单次推理生成的 Token 数。
        /// </summary>
        public int StreamingChunkSize { get; set; } = 24;

        /// <summary>
        /// 请求的音色 ID（可选）。如果设置，引擎会在推理前切换到该音色。
        /// </summary>
        public string? AvatarId { get; set; }

        /// <summary>
        /// 请求的参考音频 ID（可选）。如果设置，引擎会使用该参考音频。
        /// </summary>
        public string? ReferenceId { get; set; }

        /// <summary>
        /// G2P 优先级模式 (可选): 0-词典优先, 1-仅词典, 2-模型优先。
        /// 如果不设置，则使用全局配置。
        /// </summary>
        public int? G2PPriorityMode { get; set; }

        /// <summary>
        /// 最大重试次数。当生成的 Token 数不足时引擎会尝试重新生成。
        /// 如果不设置，则使用全局配置。
        /// </summary>
        public int? MaxRetries { get; set; }

        /// <summary>
        /// 中文最小预期 Token 系数。数值越大重试概率越高。
        /// 如果不设置，则使用全局配置。
        /// </summary>
        public float? MinTokenMultiplierChinese { get; set; }

        /// <summary>
        /// 日语最小预期 Token 系数。
        /// 如果不设置，则使用全局配置。
        /// </summary>
        public float? MinTokenMultiplierJapanese { get; set; }

        /// <summary>
        /// 英语最小预期 Token 系数。
        /// 如果不设置，则使用全局配置。
        /// </summary>
        public float? MinTokenMultiplierEnglish { get; set; }

        /// <summary>
        /// 标点/其它最小预期 Token 系数。
        /// 如果不设置，则使用全局配置。
        /// </summary>
        public float? MinTokenMultiplierOther { get; set; }

        /// <summary>
        /// 调试模式开关。
        /// 如果不设置，则使用全局配置。
        /// </summary>
        public bool? DebugMode { get; set; }

        /// <summary>
        /// 允许的语言列表 (如 ["zh", "en"])。
        /// 如果设置，将覆盖全局配置中的语种限制。
        /// </summary>
        public List<string>? Languages { get; set; }

        /// <summary>
        /// 默认配置。
        /// </summary>
        public static TtsOptions Default => new TtsOptions();
    }
}
