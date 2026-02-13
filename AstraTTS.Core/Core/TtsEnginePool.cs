using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AstraTTS.Core.Config;

namespace AstraTTS.Core.Core
{
    /// <summary>
    /// TTS 引擎池，管理并发推理实例。
    /// </summary>
    public class TtsEnginePool : IDisposable
    {
        private readonly TTSConfig _config;
        private readonly ConcurrentStack<ITtsEngine> _pool = new();
        private readonly SemaphoreSlim _semaphore;
        private ITtsEngine? _templateEngine;
        private bool _isInitialized = false;

        public TtsEnginePool(TTSConfig config)
        {
            _config = config;
            int capacity = Math.Max(1, config.PoolCapacity);
            _semaphore = new SemaphoreSlim(capacity, capacity);
        }

        /// <summary>
        /// 初始化池。加载第一个引擎作为模板。
        /// </summary>
        public async Task InitializeAsync()
        {
            if (_isInitialized) return;

            // 创建并加载第一个引擎（模板）
            _templateEngine = CreateEngine(_config);
            await _templateEngine.LoadAsync(_config);

            _pool.Push(_templateEngine);

            // 如果容量 > 1，创建更多实例
            int extraCount = _config.PoolCapacity - 1;
            if (extraCount > 0)
            {
                for (int i = 0; i < extraCount; i++)
                {
                    var engine = CreateEngine(_config);
                    if (_config.ReuseMemory)
                    {
                        // 复用模式：从模板引擎共享资源
                        ShareResources(engine, _templateEngine);
                    }
                    else
                    {
                        // 独立模式：独立加载（消耗更多内存）
                        await engine.LoadAsync(_config);
                    }
                    _pool.Push(engine);
                }
            }

            _isInitialized = true;
            Console.WriteLine($"[TtsEnginePool] Initialized with {_config.PoolCapacity} instances (ReuseMemory={_config.ReuseMemory})");
        }

        private static ITtsEngine CreateEngine(TTSConfig config)
        {
            return config.UseEngineV2 ? new TtsEngineV2() : new TtsEngineV1();
        }

        private static void ShareResources(ITtsEngine target, ITtsEngine source)
        {
            if (target is TtsEngineV1 v1Target && source is TtsEngineV1 v1Source)
            {
                v1Target.ShareResourcesFrom(v1Source);
            }
            else if (target is TtsEngineV2 v2Target && source is TtsEngineV2 v2Source)
            {
                v2Target.ShareResourcesFrom(v2Source);
            }
        }

        /// <summary>
        /// 租借一个引擎实例。使用完毕后需 Dispose 返回的 Lease。
        /// </summary>
        public async Task<EngineLease> LeaseAsync(CancellationToken cancellationToken = default)
        {
            if (!_isInitialized) await InitializeAsync();

            await _semaphore.WaitAsync(cancellationToken);

            if (_pool.TryPop(out var engine))
            {
                return new EngineLease(this, engine);
            }

            // 理论上由于信号量控制，这里不应该发生
            _semaphore.Release();
            throw new InvalidOperationException("池中无可用引擎且无法创建。");
        }

        internal void Return(ITtsEngine engine)
        {
            _pool.Push(engine);
            _semaphore.Release();
        }

        public void Dispose()
        {
            while (_pool.TryPop(out var engine))
            {
                engine.Dispose();
            }
            _semaphore.Dispose();
        }

        public int SamplingRate => _templateEngine?.SamplingRate ?? 32000;
    }

    /// <summary>
    /// 引擎租约，确保引擎在使用后归还到池中。
    /// </summary>
    public class EngineLease : IDisposable
    {
        private readonly TtsEnginePool _pool;
        private ITtsEngine? _engine;

        public ITtsEngine Engine => _engine ?? throw new ObjectDisposedException(nameof(EngineLease));

        internal EngineLease(TtsEnginePool pool, ITtsEngine engine)
        {
            _pool = pool;
            _engine = engine;
        }

        public void Dispose()
        {
            if (_engine != null)
            {
                _pool.Return(_engine);
                _engine = null;
            }
        }
    }
}
