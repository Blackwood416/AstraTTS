using System;
using System.IO;
using System.Linq;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace AstraTTS.Core.Utils
{
    public static class AudioHelper
    {
        public static float[] ReadWav(string path, int targetSampleRate)
        {
            WaveStream waveStream;
            try
            {
                // 优先使用 WaveFileReader，它是全托管实现，在 Linux 上读取标准 WAV 更稳定
                waveStream = new WaveFileReader(path);
            }
            catch
            {
                // 如果是其他格式（如 MP3）或非标准 WAV，尝试使用 AudioFileReader (Windows 依赖较重)
                waveStream = new AudioFileReader(path);
            }

            using (waveStream)
            {
                int sourceChannels = waveStream.WaveFormat.Channels;
                ISampleProvider provider = waveStream.ToSampleProvider();

                if (waveStream.WaveFormat.SampleRate != targetSampleRate)
                {
                    provider = new WdlResamplingSampleProvider(provider, targetSampleRate);
                }

                // Read all samples (Max 60s for long references)
                var read_buffer = new float[targetSampleRate * sourceChannels * 60];
                int read_count = provider.Read(read_buffer, 0, read_buffer.Length);

                if (sourceChannels == 1)
                {
                    return read_buffer.Take(read_count).ToArray();
                }

                // Downmix to mono
                int out_count = read_count / sourceChannels;
                var mono_buffer = new float[out_count];
                for (int i = 0; i < out_count; i++)
                {
                    float sum = 0;
                    for (int c = 0; c < sourceChannels; c++)
                    {
                        sum += read_buffer[i * sourceChannels + c];
                    }
                    mono_buffer[i] = sum / sourceChannels;
                }
                return mono_buffer;
            }
        }

        public static void SaveWav(string path, float[] samples, int sampleRate = 32000)
        {
            using var fs = File.Create(path);
            SaveWav(fs, samples, sampleRate);
        }

        public static void SaveWav(Stream stream, float[] samples, int sampleRate = 32000)
        {
            var format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
            using var writer = new WaveFileWriter(stream, format);
            writer.WriteSamples(samples, 0, samples.Length);
        }
    }
}
