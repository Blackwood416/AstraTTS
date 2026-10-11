namespace AstraTTS.Web.Models
{
    public class TtsRequest
    {
        /// <summary>
        /// 待合成的文本。
        /// </summary>
        public string Text { get; set; } = "";

        /// <summary>
        /// 音色 ID。不指定则使用默认音色。
        /// </summary>
        public string? AvatarId { get; set; }

        /// <summary>
        /// 参考音频 ID。不指定则使用音色的默认参考音频。
        /// </summary>
        public string? ReferenceId { get; set; }

        /// <summary>
        /// 情感 ID (可选，如 "happy"/"angry"/"sad"/"surprise"/"fearful"/"playfulness")。
        /// 需全局 UseEmotionControl=true 且 τ 文件存在。不指定或 "neutral" 表示不注入。
        /// </summary>
        public string? EmotionId { get; set; }

        /// <summary>
        /// 情感强度 α (0~1.5)。不指定则使用全局 EmotionAlpha（默认 0.3）。
        /// </summary>
        public float? EmotionAlpha { get; set; }

        /// <summary>
        /// 语速 (0.5 - 2.0)。
        /// </summary>
        public float? Speed { get; set; }

        /// <summary>
        /// 噪声系数。
        /// </summary>
        public float? NoiseScale { get; set; }

        /// <summary>
        /// 采样温度。
        /// </summary>
        public float? Temperature { get; set; }

        /// <summary>
        /// Top-K 采样。
        /// </summary>
        public int? TopK { get; set; }

        /// <summary>
        /// 流式分块大小。
        /// </summary>
        public int? StreamingChunkSize { get; set; }

        /// <summary>
        /// G2P 优先级模式: 0-词典优先, 1-仅词典, 2-模型优先。
        /// </summary>
        public int? G2PPriorityMode { get; set; }

        /// <summary>
        /// 允许的语言列表 (如 ["zh", "en"])。
        /// </summary>
        [System.Text.Json.Serialization.JsonPropertyName("languages")]
        public List<string>? Languages { get; set; }
    }
}
