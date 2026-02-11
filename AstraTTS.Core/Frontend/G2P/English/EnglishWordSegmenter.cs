using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AstraTTS.Core.Frontend.G2P.English
{
    /// <summary>
    /// 英文单词分割器，用于将长单词拆分为各独立词 (如 waterfall -> water + fall)。
    /// 辅助处理 OOV 发音。
    /// </summary>
    public class EnglishWordSegmenter
    {
        private Dictionary<string, double> _unigrams = new();
        private Dictionary<string, double> _bigrams = new();
        private const double Total = 1024908267229.0;
        private const int Limit = 24;

        public EnglishWordSegmenter(string dataDir)
        {
            LoadData(dataDir);
        }

        private void LoadData(string dataDir)
        {
            string unigramsPath = Path.Combine(dataDir, "unigrams.txt");
            string bigramsPath = Path.Combine(dataDir, "bigrams.txt");

            if (!File.Exists(unigramsPath) || !File.Exists(bigramsPath))
            {
                Console.WriteLine($"[EnglishWordSegmenter] Warning: Data files not found in {dataDir}");
                return;
            }

            try
            {
                _unigrams = LoadCounts(unigramsPath);
                _bigrams = LoadCounts(bigramsPath);
                Console.WriteLine($"[EnglishWordSegmenter] Loaded segmentation data: {_unigrams.Count} unigrams, {_bigrams.Count} bigrams.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[EnglishWordSegmenter] Error loading data: {ex.Message}");
            }
        }

        private Dictionary<string, double> LoadCounts(string path)
        {
            var dict = new Dictionary<string, double>();
            foreach (var line in File.ReadAllLines(path))
            {
                var parts = line.Split('\t');
                if (parts.Length >= 2 && double.TryParse(parts[1], out double count))
                {
                    dict[parts[0]] = count;
                }
            }
            return dict;
        }

        public string[] Segment(string text)
        {
            if (string.IsNullOrEmpty(text)) return Array.Empty<string>();

            string clean = new string(text.ToLowerInvariant().Where(c => char.IsLetterOrDigit(c)).ToArray());
            if (string.IsNullOrEmpty(clean)) return new[] { text };

            var memo = new Dictionary<(string, string), (double, List<string>)>();
            var result = Search(clean, "<s>", memo);

            // If the segmentation resulted in the same word and it's not in dictionary, 
            // the G2P will fallback to spell out anyway.
            return result.Item2.ToArray();
        }

        private double Score(string word, string? previous = null)
        {
            if (previous == null || previous == "<s>")
            {
                if (_unigrams.TryGetValue(word, out double count))
                {
                    return count / Total;
                }
                return 10.0 / (Total * Math.Pow(10, word.Length));
            }

            string bigram = $"{previous} {word}";
            if (_bigrams.TryGetValue(bigram, out double bigramCount) && _unigrams.ContainsKey(previous))
            {
                return (bigramCount / Total) / Score(previous);
            }

            return Score(word);
        }

        private (double score, List<string> words) Search(string text, string previous, Dictionary<(string, string), (double, List<string>)> memo)
        {
            if (string.IsNullOrEmpty(text))
            {
                return (0.0, new List<string>());
            }

            if (memo.TryGetValue((text, previous), out var cached)) return cached;

            double maxScore = double.NegativeInfinity;
            List<string> bestWords = new List<string>();

            for (int i = 1; i <= Math.Min(text.Length, Limit); i++)
            {
                string prefix = text.Substring(0, i);
                string suffix = text.Substring(i);

                double prefixScore = Math.Log10(Score(prefix, previous));
                var (suffixScore, suffixWords) = Search(suffix, prefix, memo);

                double totalScore = prefixScore + suffixScore;

                if (totalScore > maxScore)
                {
                    maxScore = totalScore;
                    bestWords = new List<string> { prefix };
                    bestWords.AddRange(suffixWords);
                }
            }

            memo[(text, previous)] = (maxScore, bestWords);
            return (maxScore, bestWords);
        }
    }
}
