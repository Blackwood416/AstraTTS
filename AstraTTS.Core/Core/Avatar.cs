namespace AstraTTS.Core.Core
{
    /// <summary>
    /// 代表一个可用的音色/角色配置。
    /// </summary>
    public class Avatar
    {
        /// <summary>
        /// 音色唯一标识符。
        /// </summary>
        public string Id { get; set; } = "";

        /// <summary>
        /// 音色显示名称。
        /// </summary>
        public string Name { get; set; } = "";

        /// <summary>
        /// 音色描述。
        /// </summary>
        public string? Description { get; set; }

        /// <summary>
        /// 此音色使用的引擎版本。true = V2, false = V1。
        /// 如果不指定，则使用全局配置 TTSConfig.UseEngineV2。
        /// </summary>
        public bool? UseEngineV2 { get; set; }

        /// <summary>
        /// 默认/Fallback 参考音频 ID。如果请求中不指定，则使用此音频。
        /// </summary>
        public string DefaultReferenceId { get; set; } = "default";

        /// <summary>
        /// 此音色可用的所有参考音频。
        /// </summary>
        public List<ReferenceAudio> References { get; set; } = new();

        /// <summary>
        /// 获取默认参考音频。
        /// </summary>
        public ReferenceAudio? GetDefaultReference()
        {
            return References.FirstOrDefault(r => r.Id == DefaultReferenceId)
                   ?? References.FirstOrDefault();
        }

        /// <summary>
        /// 获取指定 ID 的参考音频，如果找不到则返回默认音频。
        /// </summary>
        public ReferenceAudio? GetReference(string? referenceId)
        {
            if (string.IsNullOrEmpty(referenceId))
                return GetDefaultReference();

            return References.FirstOrDefault(r => r.Id == referenceId)
                   ?? GetDefaultReference();
        }
    }

    /// <summary>
    /// 代表一个参考音频配置。
    /// </summary>
    public class ReferenceAudio
    {
        /// <summary>
        /// 参考音频唯一标识符 (在 Avatar 内唯一)。
        /// </summary>
        public string Id { get; set; } = "default";

        /// <summary>
        /// 参考音频显示名称。
        /// </summary>
        public string? Name { get; set; }

        /// <summary>
        /// 参考音频路径。相对路径会自动拼接 Avatar 的 references 目录。
        /// 例如: "normal.wav" -> "{AvatarsDir}/{avatarId}/references/normal.wav"
        /// </summary>
        public string AudioPath { get; set; } = "";

        /// <summary>
        /// 参考音频对应的文本内容。
        /// </summary>
        public string Text { get; set; } = "";

        /// <summary>
        /// 语言标记 (可选)。
        /// </summary>
        public string? Language { get; set; }

        /// <summary>
        /// 获取完整的音频路径。如果是相对路径，则拼接 Avatar 的 references 目录。
        /// </summary>
        /// <param name="avatarsDir">音色目录根路径</param>
        /// <param name="avatarId">音色 ID</param>
        public string GetFullAudioPath(string avatarsDir, string avatarId)
        {
            if (string.IsNullOrEmpty(AudioPath))
                return "";

            // 1. 如果是绝对路径，直接返回
            if (Path.IsPathRooted(AudioPath))
                return AudioPath;

            // 2. 检查是否已经是相对于当前目录的可访问路径
            if (File.Exists(AudioPath))
                return Path.GetFullPath(AudioPath);

            // 3. 默认行为：拼接 {avatarsDir}/{avatarId}/references/{AudioPath}
            return Path.Combine(avatarsDir, avatarId, "references", AudioPath);
        }
    }
}
