using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace AstraTTS.Core.Frontend.G2P.English
{
    /// <summary>
    /// 英文词性标注器，移植自 NLTK 的 Averaged Perceptron Tagger。
    /// 用于解决同形异义词 (Homographs) 的发音歧义。
    /// </summary>
    public class EnglishPosTagger
    {
        private Dictionary<string, Dictionary<string, float>> _weights = new();
        private Dictionary<string, string> _tagDict = new();
        private HashSet<string> _classes = new();

        private static readonly string[] StartTokens = { "-START-", "-START2-" };
        private static readonly string[] EndTokens = { "-END-", "-END2-" };

        public EnglishPosTagger(string modelDir)
        {
            LoadModel(modelDir);
        }

        private void LoadModel(string modelDir)
        {
            string weightsPath = Path.Combine(modelDir, "averaged_perceptron_tagger_eng.weights.json");
            string tagDictPath = Path.Combine(modelDir, "averaged_perceptron_tagger_eng.tagdict.json");
            string classesPath = Path.Combine(modelDir, "averaged_perceptron_tagger_eng.classes.json");

            if (!File.Exists(weightsPath) || !File.Exists(tagDictPath) || !File.Exists(classesPath))
            {
                Console.WriteLine($"[EnglishPosTagger] Warning: POS model files not found in {modelDir}");
                return;
            }

            try
            {
                var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };

                _weights = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, float>>>(File.ReadAllText(weightsPath), options) ?? new();
                _tagDict = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(tagDictPath), options) ?? new();
                var classList = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(classesPath), options) ?? new();
                _classes = new HashSet<string>(classList);

                Console.WriteLine($"[EnglishPosTagger] Loaded POS model: {_weights.Count} features, {_tagDict.Count} tagdict entries.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[EnglishPosTagger] Error loading model: {ex.Message}");
            }
        }

        public string[] Tag(string[] tokens)
        {
            if (tokens == null || tokens.Length == 0) return Array.Empty<string>();

            int n = tokens.Length;
            string[] tags = new string[n];
            string prev = StartTokens[0];
            string prev2 = StartTokens[1];

            // Context: START + NormalizedWords + END
            string[] context = new string[n + 4];
            context[0] = StartTokens[0];
            context[1] = StartTokens[1];
            for (int j = 0; j < n; j++) context[j + 2] = Normalize(tokens[j]);
            context[n + 2] = EndTokens[0];
            context[n + 3] = EndTokens[1];

            for (int i = 0; i < n; i++)
            {
                string word = tokens[i];
                string tag;
                if (_tagDict.TryGetValue(word, out var dictTag))
                {
                    tag = dictTag;
                }
                else
                {
                    var features = GetFeatures(i, word, context, prev, prev2);
                    tag = Predict(features);
                }

                tags[i] = tag;

                prev2 = prev;
                prev = tag;
            }

            RefineTags(tokens, tags);

            return tags;
        }

        /// <summary>
        /// 对标注结果进行启发式微调，修正模型权重导致的常见错误 (如 read + yesterday)。
        /// </summary>
        private void RefineTags(string[] tokens, string[] tags)
        {
            // 简单启发式：如果在句子中发现了过去式时间状语，则将标注为 VBP (或 VB) 的 "read" 修正为 VBD
            string[] pastMarkers = { "yesterday", "ago", "last", "previously", "before" };
            bool hasPastMarker = tokens.Any(t => pastMarkers.Any(pm => t.Contains(pm, StringComparison.OrdinalIgnoreCase)));

            if (hasPastMarker)
            {
                for (int i = 0; i < tokens.Length; i++)
                {
                    if (string.Equals(tokens[i], "read", StringComparison.OrdinalIgnoreCase))
                    {
                        // 如果模型将其标为 VBP (一般由于主语是 I/You/We/They 导致权重偏见)，
                        // 但由于有过去式状语，强制修正为 VBD
                        if (tags[i] == "VBP" || tags[i] == "VB")
                        {
                            tags[i] = "VBD";
                        }
                    }
                }
            }
        }

        private string Normalize(string word)
        {
            if (string.IsNullOrEmpty(word)) return word;
            if (word.Contains("-") && word[0] != '-') return "!HYPHEN";
            if (word.Length == 4 && word.All(char.IsDigit)) return "!YEAR";
            if (char.IsDigit(word[0])) return "!DIGITS";
            return word.ToLowerInvariant();
        }

        private string Predict(Dictionary<string, int> features)
        {
            var scores = new Dictionary<string, double>();

            foreach (var feat in features.Keys)
            {
                if (!_weights.TryGetValue(feat, out var weights)) continue;

                int value = features[feat];
                foreach (var label in weights.Keys)
                {
                    double contrib = value * (double)weights[label];
                    scores[label] = scores.GetValueOrDefault(label, 0) + contrib;
                }
            }

            if (scores.Count == 0) return _classes.FirstOrDefault() ?? "NN";

            // NLTK's secondary sort is alphabetic to ensure consistency
            return scores.OrderByDescending(kv => kv.Value)
                         .ThenBy(kv => kv.Key)
                         .First().Key;
        }

        private Dictionary<string, int> GetFeatures(int i, string word, string[] context, string prev, string prev2)
        {
            var features = new Dictionary<string, int>();
            i += 2; // Offset for START tokens

            void Add(params string[] parts)
            {
                string feat = string.Join(" ", parts);
                features[feat] = features.GetValueOrDefault(feat, 0) + 1;
            }

            Add("bias");
            Add("i suffix", word.Length >= 3 ? word[^3..] : word);
            Add("i pref1", word.Length > 0 ? word[0].ToString() : "");
            Add("i-1 tag", prev);
            Add("i-2 tag", prev2);
            Add("i tag+i-2 tag", prev, prev2);
            Add("i word", context[i]);
            Add("i-1 tag+i word", prev, context[i]);
            Add("i-1 word", context[i - 1]);
            Add("i-1 suffix", context[i - 1].Length >= 3 ? context[i - 1][^3..] : context[i - 1]);
            Add("i-2 word", context[i - 2]);
            Add("i+1 word", context[i + 1]);
            Add("i+1 suffix", context[i + 1].Length >= 3 ? context[i + 1][^3..] : context[i + 1]);
            Add("i+2 word", context[i + 2]);

            return features;
        }
    }
}
