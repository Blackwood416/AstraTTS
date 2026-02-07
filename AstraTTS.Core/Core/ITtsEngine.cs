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
        /// 执行异步推理并返回完整的音频数据。
        /// </summary>
        /// <param name="text">待生成的文本。</param>
        /// <param name="options">推理选项（如语速、噪声比例等）。</param>
        /// <returns>采样率为 32kHz 的音频样本数组。</returns>
        Task<float[]> PredictAsync(string text, TtsOptions options);

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
        /// 流式输出的 Chunk 大小（以 Token 数计算）。
        /// </summary>
        public int StreamingChunkSize { get; set; } = 24;

        /// <summary>
        /// 流式推理中的单个 Chunk Token 数量（V2 专用）。
        /// </summary>
        public int StreamingChunkTokens { get; set; } = 24;

        /// <summary>
        /// 请求的音色 ID（可选）。如果设置，引擎会在推理前切换到该音色。
        /// </summary>
        public string? AvatarId { get; set; }

        /// <summary>
        /// 请求的参考音频 ID（可选）。如果设置，引擎会使用该参考音频。
        /// </summary>
        public string? ReferenceId { get; set; }

        /// <summary>
        /// 默认配置。
        /// </summary>
        public static TtsOptions Default => new TtsOptions();
    }
}
