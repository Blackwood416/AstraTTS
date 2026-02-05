using System;

namespace AstraTTS.Core.Utils
{
    /// <summary>
    /// 简单一阶高通滤波器，消除直流偏移
    /// </summary>
    public class HighPassFilter
    {
        private float _prevInput;
        private float _prevOutput;
        private readonly float _alpha;

        public HighPassFilter(float sampleRate, float cutoffHz = 20f)
        {
            float rc = 1f / (2f * MathF.PI * cutoffHz);
            float dt = 1f / sampleRate;
            _alpha = rc / (rc + dt);
        }

        public void Process(Span<float> samples)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                float input = samples[i];
                float output = _alpha * (_prevOutput + input - _prevInput);
                _prevInput = input;
                _prevOutput = output;
                samples[i] = output;
            }
        }
    }
}
