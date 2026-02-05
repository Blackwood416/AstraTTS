using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AstraTTS.Core.Config;

namespace AstraTTS.Core.Core
{
    /// <summary>
    /// AstraTTS.Core SDK 核心类，提供高层级 API。
    /// </summary>
    public class AstraTtsSdk : IDisposable
    {
        private ITtsEngine _engine;
        private TTSConfig _config;
        private readonly object _lock = new();

        /// <summary>
        /// 获取采样率。
        /// </summary>
        public int SamplingRate => _engine.SamplingRate;

        /// <summary>
        /// 获取当前生效的配置。
        /// </summary>
        public TTSConfig Config => _config;

        /// <summary>
        /// 获取所有可用的音色。
        /// </summary>
        public IReadOnlyList<Avatar> Avatars => _config.Avatars;

        /// <summary>
        /// 初始化 SDK。
        /// </summary>
        /// <param name="config">配置对象。如果为 null 则尝试加载默认配置。</param>
        public AstraTtsSdk(TTSConfig? config = null)
        {
            _config = config ?? TTSConfig.Load();
            _engine = CreateEngine(_config);
        }

        private static ITtsEngine CreateEngine(TTSConfig config)
        {
            return config.UseEngineV2 ? new TtsEngineV2() : new TtsEngineV1();
        }

        /// <summary>
        /// 异步初始化引擎。
        /// </summary>
        public async Task InitializeAsync()
        {
            await _engine.LoadAsync(_config);
        }

        /// <summary>
        /// 热重载配置并重新初始化引擎。
        /// </summary>
        public async Task ReloadConfigAsync()
        {
            var newConfig = TTSConfig.Reload();
            var newEngine = CreateEngine(newConfig);
            await newEngine.LoadAsync(newConfig);

            lock (_lock)
            {
                var oldEngine = _engine;
                _engine = newEngine;
                _config = newConfig;
                oldEngine.Dispose();
            }

            Console.WriteLine("[AstraTTS] Configuration reloaded successfully.");
        }

        /// <summary>
        /// 获取指定 ID 的音色。
        /// </summary>
        public Avatar? GetAvatar(string? avatarId)
        {
            if (string.IsNullOrEmpty(avatarId))
                avatarId = _config.DefaultAvatarId;

            return _config.Avatars.Find(a => a.Id == avatarId);
        }

        /// <summary>
        /// 全量合成音频。
        /// </summary>
        /// <param name="text">待合成文本</param>
        /// <param name="options">推理选项。如果为 null 则使用配置中的默认值。</param>
        /// <param name="avatarId">音色 ID (可选)。</param>
        /// <param name="referenceId">参考音频 ID (可选)。</param>
        /// <returns>音频采样数据 (PCM Float32)</returns>
        public async Task<float[]> PredictAsync(string text, TtsOptions? options = null, string? avatarId = null, string? referenceId = null)
        {
            var opt = options ?? GetDefaultOptions();

            // Avatar 和 Reference 处理 (待引擎支持后集成)
            // var avatar = GetAvatar(avatarId);
            // var reference = avatar?.GetReference(referenceId);

            return await _engine.PredictAsync(text, opt);
        }

        /// <summary>
        /// 流式合成音频。
        /// </summary>
        /// <param name="text">待合成文本</param>
        /// <param name="options">推理选项。如果为 null 则使用配置中的默认值。</param>
        /// <param name="avatarId">音色 ID (可选)。</param>
        /// <param name="referenceId">参考音频 ID (可选)。</param>
        /// <param name="cancellationToken">取消令牌。</param>
        /// <returns>异步音频块流</returns>
        public IAsyncEnumerable<float[]> PredictStreamAsync(string text, TtsOptions? options = null, string? avatarId = null, string? referenceId = null, CancellationToken cancellationToken = default)
        {
            var opt = options ?? GetDefaultOptions();
            return _engine.PredictStreamAsync(text, opt, cancellationToken);
        }

        private TtsOptions GetDefaultOptions()
        {
            return new TtsOptions
            {
                Speed = _config.Speed,
                NoiseScale = _config.NoiseScale,
                Temperature = _config.Temperature,
                TopK = _config.TopK,
                StreamingChunkSize = _config.StreamingChunkSize,
                StreamingChunkTokens = _config.StreamingChunkTokens
            };
        }

        public void Dispose()
        {
            _engine.Dispose();
        }
    }
}
