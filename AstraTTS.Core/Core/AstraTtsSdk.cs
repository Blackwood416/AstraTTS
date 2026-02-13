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
        private TtsEnginePool _pool;
        private TTSConfig _config;
        private readonly object _lock = new();

        /// <summary>
        /// 获取采样率。
        /// </summary>
        public int SamplingRate => _pool.SamplingRate;

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
            _pool = new TtsEnginePool(_config);
        }

        /// <summary>
        /// 异步初始化引擎。
        /// </summary>
        public async Task InitializeAsync()
        {
            await _pool.InitializeAsync();
        }

        /// <summary>
        /// 重新加载配置并重新初始化引擎。
        /// </summary>
        public async Task ReloadConfigAsync()
        {
            var newConfig = TTSConfig.Reload();
            await ApplyConfigAsync(newConfig);
            Console.WriteLine("[AstraTTS] Configuration reloaded successfully.");
        }

        /// <summary>
        /// 直接应用当前配置（或指定配置）到引擎。
        /// 不会从磁盘重新加载配置文件。
        /// </summary>
        public async Task ApplyConfigAsync(TTSConfig? config = null)
        {
            var targetConfig = config ?? _config;
            var newPool = new TtsEnginePool(targetConfig);
            await newPool.InitializeAsync();

            lock (_lock)
            {
                var oldPool = _pool;
                _pool = newPool;
                _config = targetConfig;
                oldPool.Dispose();
            }
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
        /// <returns>包含音频数据和元数据的 TtsResult</returns>
        public async Task<TtsResult> PredictAsync(string text, TtsOptions? options = null, string? avatarId = null, string? referenceId = null)
        {
            var opt = options ?? GetDefaultOptions();
            opt.AvatarId = avatarId ?? _config.DefaultAvatarId;
            opt.ReferenceId = referenceId;
            opt.Languages ??= _config.G2P.Languages != null ? new List<string>(_config.G2P.Languages) : null;

            using var lease = await _pool.LeaseAsync();
            return await lease.Engine.PredictAsync(text, opt);
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
        public async IAsyncEnumerable<float[]> PredictStreamAsync(string text, TtsOptions? options = null, string? avatarId = null, string? referenceId = null, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var opt = options ?? GetDefaultOptions();
            opt.AvatarId = avatarId ?? _config.DefaultAvatarId;
            opt.ReferenceId = referenceId;
            opt.Languages ??= _config.G2P.Languages != null ? new List<string>(_config.G2P.Languages) : null;

            using var lease = await _pool.LeaseAsync(cancellationToken);
            await foreach (var chunk in lease.Engine.PredictStreamAsync(text, opt, cancellationToken))
            {
                yield return chunk;
            }
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
                Languages = _config.G2P.Languages != null ? new List<string>(_config.G2P.Languages) : null
            };
        }

        public void Dispose()
        {
            _pool.Dispose();
        }
    }
}
