using System;
using System.Buffers;
using System.Runtime.InteropServices;
using AstraTTS.Core.Utils;

namespace AstraTTS.Core.Core
{
    /// <summary>
    /// 音频后处理流水线。
    /// 集成了 CrossFade、高通滤波、PCM 转换等功能，旨在实现零 GC 运行。
    /// </summary>
    public class AudioPipeline : IDisposable
    {
        private readonly int _sampleRate;
        private readonly int _crossFadeSamples;
        private readonly HighPassFilter _hpFilter;

        private float[]? _prevChunkTail;
        private int _prevTailLen;

        private readonly object _lock = new();

        public AudioPipeline(int sampleRate = 32000, int crossFadeMs = 20, int hpCutoff = 20)
        {
            _sampleRate = sampleRate;
            _crossFadeSamples = (sampleRate * crossFadeMs) / 1000;
            _hpFilter = new HighPassFilter(sampleRate, hpCutoff);

            // 预分配用于 CrossFade 的尾部缓冲区
            _prevChunkTail = ArrayPool<float>.Shared.Rent(_crossFadeSamples);
            _prevTailLen = 0;
        }

        /// <summary>
        /// 处理一个音频块。
        /// </summary>
        /// <param name="input">输入音频 (float[])</param>
        /// <param name="isFinal">是否为最后一块</param>
        /// <returns>处理后租用的 byte[] 缓冲区 (PCM16)。调用方负责返回 ArrayPool。</returns>
        public (byte[] pcmBuffer, int pcmLength) ProcessChunk(float[] input, int length, bool isFinal)
        {
            lock (_lock)
            {
                // 1. 应用高通滤波 (原地处理)
                _hpFilter.Process(input.AsSpan(0, length));

                // 2. CrossFade 处理
                int effectiveLen = length;

                // 与前一块的尾部合并
                if (_prevTailLen > 0 && effectiveLen > _crossFadeSamples)
                {
                    for (int i = 0; i < _crossFadeSamples; i++)
                    {
                        float alpha = (float)i / _crossFadeSamples;
                        input[i] = _prevChunkTail![i] * (1f - alpha) + input[i] * alpha;
                    }
                }

                // 保存当前块的尾部用于下次 CrossFade
                if (!isFinal && effectiveLen > _crossFadeSamples)
                {
                    Array.Copy(input, effectiveLen - _crossFadeSamples, _prevChunkTail!, 0, _crossFadeSamples);
                    _prevTailLen = _crossFadeSamples;
                    effectiveLen -= _crossFadeSamples; // 隐藏尾部，留到下一块播放
                }
                else
                {
                    // 最后一块：应用淡出并重置
                    if (isFinal && effectiveLen > _crossFadeSamples)
                    {
                        int start = effectiveLen - _crossFadeSamples;
                        for (int i = 0; i < _crossFadeSamples; i++)
                        {
                            float fadeOut = 1f - (float)i / _crossFadeSamples;
                            input[start + i] *= fadeOut;
                        }
                    }
                    _prevTailLen = 0;
                }

                // 3. 转换为 PCM16
                byte[] pcmBuf = ArrayPool<byte>.Shared.Rent(effectiveLen * 2);
                var pcmShorts = MemoryMarshal.Cast<byte, short>(pcmBuf.AsSpan(0, effectiveLen * 2));

                for (int i = 0; i < effectiveLen; i++)
                {
                    pcmShorts[i] = (short)(Math.Clamp(input[i], -1f, 1f) * 32767);
                }

                return (pcmBuf, effectiveLen * 2);
            }
        }

        public void Reset()
        {
            lock (_lock)
            {
                _prevTailLen = 0;
            }
        }

        public void Dispose()
        {
            if (_prevChunkTail != null)
            {
                ArrayPool<float>.Shared.Return(_prevChunkTail);
                _prevChunkTail = null;
            }
        }
    }
}
