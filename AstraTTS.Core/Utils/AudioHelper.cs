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
            using var reader = new AudioFileReader(path);
            ISampleProvider provider = reader;

            if (reader.WaveFormat.SampleRate != targetSampleRate)
            {
                provider = new WdlResamplingSampleProvider(reader, targetSampleRate);
            }

            // Read all samples (Max 30s)
            var buffer = new float[targetSampleRate * 30];
            int read_count = provider.Read(buffer, 0, buffer.Length);

            return buffer.Take(read_count).ToArray();
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
