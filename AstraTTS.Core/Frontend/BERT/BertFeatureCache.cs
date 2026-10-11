using System;
using System.Collections.Generic;
using System.Linq;

namespace AstraTTS.Core.Frontend.BERT
{
    /// <summary>
    /// BERT 特征缓存的 FIFO 实现（线程安全）。
    ///
    /// 动机：`RobertaFeatureExtractor.Extract` 每次调用约 280–390 ms（其中约 278 ms 是
    /// 图中 Loop 节点的 CPU 控制流）。同一文本（且 word2ph 相同）的输出完全确定，
    /// 而典型使用场景是同一参考文本/短语被反复合成，因此值得缓存。
    ///
    /// 语义约束：
    ///   - 缓存的是 `Extract` 的返回值 `float[rows, hidden]`，**调用方必须只读**
    ///     （V1/V2 现状均为 Buffer.BlockCopy 后即弃，不修改）。
    ///   - 键包含 word2ph，避免同一文本在不同分词/对齐下被错误命中。
    ///   - 淘汰策略：FIFO，容量固定，避免无界增长。
    /// </summary>
    public sealed class BertFeatureCache
    {
        private readonly int _capacity;
        private readonly Dictionary<string, float[,]> _map;
        private readonly Queue<string> _order = new Queue<string>();
        private readonly object _lock = new object();

        private long _hits;
        private long _misses;

        public BertFeatureCache(int capacity = 256)
        {
            if (capacity < 1) capacity = 1;
            _capacity = capacity;
            _map = new Dictionary<string, float[,]>(capacity);
        }

        /// <summary>缓存容量。</summary>
        public int Capacity => _capacity;

        /// <summary>命中次数（仅统计）。</summary>
        public long Hits => System.Threading.Interlocked.Read(ref _hits);

        /// <summary>未命中次数（仅统计）。</summary>
        public long Misses => System.Threading.Interlocked.Read(ref _misses);

        /// <summary>构造缓存键。word2ph 参与键，避免不同对齐复用同一结果。</summary>
        public static string MakeKey(string text, int[] word2ph)
        {
            // 使用不会出现在正常文本里的分隔符，避免拼接歧义
            return text + "\u0001" + string.Join("\u0002", word2ph ?? Array.Empty<int>());
        }

        /// <summary>尝试取缓存。命中返回 true，value 为只读共享数组。</summary>
        public bool TryGet(string key, out float[,] value)
        {
            lock (_lock)
            {
                if (_map.TryGetValue(key, out value))
                {
                    System.Threading.Interlocked.Increment(ref _hits);
                    return true;
                }
            }
            value = null!;
            System.Threading.Interlocked.Increment(ref _misses);
            return false;
        }

        /// <summary>写入缓存（超出容量时按 FIFO 淘汰）。</summary>
        public void Set(string key, float[,] value)
        {
            lock (_lock)
            {
                if (_map.ContainsKey(key)) return;
                if (_map.Count >= _capacity)
                {
                    // FIFO：淘汰最早写入的键
                    while (_order.Count > 0)
                    {
                        var oldest = _order.Dequeue();
                        if (_map.Remove(oldest)) break;
                    }
                }
                _map[key] = value;
                _order.Enqueue(key);
            }
        }

        /// <summary>清空缓存（例如配置重载后）。</summary>
        public void Clear()
        {
            lock (_lock)
            {
                _map.Clear();
                _order.Clear();
            }
        }
    }
}
