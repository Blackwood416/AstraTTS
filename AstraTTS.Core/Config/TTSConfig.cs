using System.Text.Json.Serialization;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using AstraTTS.Core.Core;

namespace AstraTTS.Core.Config
{
    public class TTSConfig
    {
        // ============================================================
        // 资源目录配置 (核心路径 - 只需配置这个)
        // ============================================================

        private string _resourcesDir = "resources";
        public string ResourcesDir
        {
            get => Path.IsPathRooted(_resourcesDir)
                ? _resourcesDir
                : Path.Combine(AppDomain.CurrentDomain.BaseDirectory, _resourcesDir);
            set => _resourcesDir = value;
        }

        // ============================================================
        // 派生路径 (自动计算，无需手动配置)
        // ============================================================

        /// <summary>
        /// 共享资源目录 (字典等)
        /// </summary>
        [JsonIgnore]
        public string SharedDir => Path.Combine(ResourcesDir, "shared");

        /// <summary>
        /// 字典目录
        /// </summary>
        [JsonIgnore]
        public string DictionariesDir => Path.Combine(SharedDir, "dictionaries");

        /// <summary>
        /// V1 引擎共享资源目录 (BERT, HuBERT 等)
        /// </summary>
        [JsonIgnore]
        public string V1ExtraDir => Path.Combine(SharedDir, "v1_extra");

        /// <summary>
        /// V1 引擎音色特定模型目录
        /// </summary>
        [JsonIgnore]
        public string ModelsV1BaseDir => Path.Combine(ResourcesDir, "models_v1");

        /// <summary>
        /// V2 引擎模型基础目录
        /// </summary>
        [JsonIgnore]
        public string ModelsV2BaseDir => Path.Combine(ResourcesDir, "models_v2");

        /// <summary>
        /// 获取指定 Avatar 的 V1 模型目录
        /// </summary>
        public string GetModelsV1Dir(string avatarId) => Path.Combine(ModelsV1BaseDir, avatarId);

        /// <summary>
        /// 获取指定 Avatar 的 V2 模型目录
        /// </summary>
        public string GetModelsV2Dir(string avatarId) => Path.Combine(ModelsV2BaseDir, avatarId);

        /// <summary>
        /// 获取默认 Avatar 的 V1 模型目录 (兼容旧代码)
        /// </summary>
        [JsonIgnore]
        public string ModelsV1Dir => GetModelsV1Dir(DefaultAvatarId);

        /// <summary>
        /// 获取默认 Avatar 的 V2 模型目录 (兼容旧代码)
        /// </summary>
        [JsonIgnore]
        public string ModelsV2Dir => GetModelsV2Dir(DefaultAvatarId);

        // ============================================================
        // V1 引擎模型路径
        // ============================================================

        [JsonIgnore]
        public string HubertPath => Path.Combine(V1ExtraDir, "hubert", "chinese-hubert-base_full.onnx");

        [JsonIgnore]
        public string SpeakerEncoderPath => Path.Combine(V1ExtraDir, "speaker_encoder.onnx");

        // V1 音色特定的 VITS 模型目录 (核心模型直接放置在音色目录下)
        [JsonIgnore]
        public string V1TtsDir => ModelsV1Dir;

        [JsonIgnore]
        public string BertModelPath => Path.Combine(V1ExtraDir, "bert", "roberta.onnx");

        [JsonIgnore]
        public string TokenizerJsonPath => Path.Combine(V1ExtraDir, "bert", "tokenizer", "tokenizer.json");

        // ============================================================
        // 按 Avatar 获取 V1 路径
        // ============================================================

        // public string GetHubertPath(string avatarId) => HubertPath;
        // public string GetSpeakerEncoderPath(string avatarId) => SpeakerEncoderPath;
        // public string GetV1TtsDir(string avatarId) => GetModelsV1Dir(avatarId);
        // public string GetBertModelPath(string avatarId) => BertModelPath;
        // public string GetTokenizerJsonPath(string avatarId) => TokenizerJsonPath;

        // ============================================================
        // 共享资源路径 (G2P 字典与模型)
        // ============================================================

        [JsonIgnore]
        public string G2PDir => Path.Combine(SharedDir, "g2p");

        [JsonIgnore]
        public string G2PDictsDir => Path.Combine(G2PDir, "dicts");

        [JsonIgnore]
        public string G2PModelsDir => Path.Combine(G2PDir, "models");

        [JsonIgnore]
        public string CmuDict => Path.Combine(G2PDictsDir, "english", "cmudict.dict");

        [JsonIgnore]
        public string PinyinDict => Path.Combine(G2PDictsDir, "chinese", "mandarin_pinyin.dict");

        [JsonIgnore]
        public string ChineseG2PDict => Path.Combine(G2PDictsDir, "chinese", "opencpop-strict.txt");

        [JsonIgnore]
        public string PolyphonicJson => Path.Combine(G2PDictsDir, "chinese", "polyphonic.json");

        [JsonIgnore]
        public string NeuralG2PModel => Path.Combine(G2PModelsDir, "checkpoint20.npz");

        [JsonIgnore]
        public string EnglishSpecialDict => Path.Combine(G2PDictsDir, "english", "en_special_words.txt");

        /// <summary>
        /// 英文词性标注模型目录
        /// </summary>
        [JsonIgnore]
        public string EnglishPosTaggerDir => Path.Combine(G2PModelsDir, "english", "taggers", "averaged_perceptron_tagger_eng");

        /// <summary>
        /// 英文单词分割数据目录
        /// </summary>
        [JsonIgnore]
        public string EnglishWordSegmentDir => Path.Combine(G2PModelsDir, "english", "wordsegment");

        /// <summary>
        /// 日语 OpenJTalk 词典目录
        /// </summary>
        [JsonIgnore]
        public string JapaneseDictDir => Path.Combine(G2PDictsDir, "japanese", "open_jtalk_dic_utf_8-1.11");

        [JsonIgnore]
        public string JiebaDictDir => Path.Combine(G2PDictsDir, "chinese", "jieba");

        // ============================================================
        // 音色 (Avatar) 配置
        // ============================================================

        /// <summary>
        /// 默认音色 ID。如果请求中不指定，则使用此音色。
        /// </summary>
        public string DefaultAvatarId { get; set; } = "default";

        /// <summary>
        /// 音色目录。
        /// </summary>
        [JsonIgnore]
        public string AvatarsDir => Path.Combine(ResourcesDir, "avatars");

        /// <summary>
        /// 音色列表。可以在配置文件中手动定义，也可以从 AvatarsDir 自动扫描。
        /// </summary>
        public List<Avatar> Avatars { get; set; } = new();

        /// <summary>
        /// 获取默认 Avatar
        /// </summary>
        public Avatar? GetDefaultAvatar()
        {
            return Avatars.FirstOrDefault(a => a.Id == DefaultAvatarId) ?? Avatars.FirstOrDefault();
        }

        /// <summary>
        /// 获取默认参考音频的完整路径和文本
        /// </summary>
        // public (string audioPath, string text)? GetDefaultReferenceAudio()
        // {
        //     var avatar = GetDefaultAvatar();
        //     if (avatar == null) return null;

        //     var reference = avatar.GetDefaultReference();
        //     if (reference == null) return null;

        //     var fullPath = reference.GetFullAudioPath(AvatarsDir, avatar.Id);
        //     return (fullPath, reference.Text);
        // }

        // ============================================================
        // 引擎选择
        // ============================================================

        /// <summary>
        /// 使用 V2 推理引擎 (基于 GPT-SoVITS minimal inference)
        /// </summary>
        public bool UseEngineV2 { get; set; } = false;

        // ============================================================
        // 硬件加速
        // ============================================================

        public bool UseDirectML { get; set; } = false;

        // ============================================================
        // 性能配置
        // ============================================================

        /// <summary>
        /// ONNX Runtime 内部操作并行线程数。0 表示使用默认值。
        /// </summary>
        public int IntraOpNumThreads { get; set; } = 0;

        /// <summary>
        /// ONNX Runtime 跨操作并行线程数。0 表示使用默认值。
        /// </summary>
        public int InterOpNumThreads { get; set; } = 0;

        /// <summary>
        /// 图优化级别 (0=禁用, 1=基础, 2=激进)。
        /// </summary>
        public int GraphOptimizationLevel { get; set; } = 1;

        /// <summary>
        /// 推理池容量（并发数）。默认为 1。
        /// </summary>
        public int PoolCapacity { get; set; } = 1;

        /// <summary>
        /// 是否复用内存（共享模型 Session）。默认为 true。
        /// </summary>
        public bool ReuseMemory { get; set; } = true;

        // ============================================================
        // 合成参数
        // ============================================================

        /// <summary>
        /// 语速调节 (0.5 - 2.0)
        /// </summary>
        public float Speed { get; set; } = 1.0f;

        /// <summary>
        /// 噪声系数 (影响音色变化)
        /// </summary>
        public float NoiseScale { get; set; } = 0.35f;

        /// <summary>
        /// Top-K 采样 (推荐 15-50)
        /// </summary>
        public int TopK { get; set; } = 15;

        /// <summary>
        /// 采样温度 (越高越随机)
        /// </summary>
        public float Temperature { get; set; } = 1.0f;

        // ============================================================
        // 流式配置
        // ============================================================

        public int StreamingChunkSize { get; set; } = 22;
        public int StreamingPreBufferChunks { get; set; } = 2;
        public bool StreamingMode { get; set; } = true;

        // ============================================================
        // 音频后端
        // ============================================================

        public bool WasapiExclusiveMode { get; set; } = true;
        public int LockFreeBufferSize { get; set; } = 65536 * 32;

        // ============================================================
        // 调试与推理配置
        // ============================================================

        /// <summary>
        /// 是否启用调试模式。启用后会输出详细的推理日志。
        /// </summary>
        public bool DebugMode { get; set; } = false;

        public InferenceConfig Inference { get; set; } = new InferenceConfig();

        public class InferenceConfig
        {
            /// <summary>
            /// 最大重试次数。当生成的 Token 数不足时引擎会尝试重新生成。
            /// </summary>
            public int MaxRetries { get; set; } = 3;

            /// <summary>
            /// 中文最小预期 Token 系数 (默认 2.1)。数值越大重试概率越高。
            /// </summary>
            public float MinTokenMultiplierChinese { get; set; } = 2.1f;

            /// <summary>
            /// 日语最小预期 Token 系数 (默认 2.0)。
            /// </summary>
            public float MinTokenMultiplierJapanese { get; set; } = 2.0f;

            /// <summary>
            /// 英语最小预期 Token 系数 (默认 1.3)。
            /// </summary>
            public float MinTokenMultiplierEnglish { get; set; } = 1.3f;

            /// <summary>
            /// 标点/其它最小预期 Token 系数 (默认 0.5)。
            /// </summary>
            public float MinTokenMultiplierOther { get; set; } = 0.5f;
        }

        // ============================================================
        // G2P 配置
        // ============================================================

        public G2PConfig G2P { get; set; } = new G2PConfig();

        public class G2PConfig
        {
            /// <summary>
            /// 用户自定义词典路径 (相对于 SharedDir)
            /// </summary>
            public string? CustomDictPath { get; set; } = "custom_dict.txt";

            /// <summary>
            /// 优先级模式：0-词典优先, 1-仅词典, 2-模型优先
            /// </summary>
            public int PriorityMode { get; set; } = 0;

            /// <summary>
            /// 允许的语言列表 (最多 2 种)。可选值: "zh", "en", "ja"
            /// </summary>
            public List<string> Languages { get; set; } = new() { "zh", "en" };
        }

        // ============================================================
        // 获取完整的自定义词典路径
        // ============================================================

        [JsonIgnore]
        public string? CustomDictFullPath =>
            string.IsNullOrEmpty(G2P.CustomDictPath) ? null : Path.Combine(SharedDir, G2P.CustomDictPath);

        // ============================================================
        // 配置加载与保存
        // ============================================================

        private static string? _loadedPath;

        [JsonIgnore]
        public static string? LoadedPath => _loadedPath;

        [JsonIgnore]
        [YamlIgnore]
        public static string DefaultConfigName => "config.yaml";

        [JsonIgnore]
        [YamlIgnore]
        public static string LegacyConfigName => "config.json";

        public static string DefaultConfigPath => Path.Combine(AppDomain.CurrentDomain.BaseDirectory, DefaultConfigName);

        public static TTSConfig Load()
        {
            // Try CWD first
            if (File.Exists(DefaultConfigName)) return Load(DefaultConfigName);
            if (File.Exists(LegacyConfigName)) return Load(LegacyConfigName);

            // Then try AppDir
            var appDirDefault = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, DefaultConfigName);
            if (File.Exists(appDirDefault)) return Load(appDirDefault);

            var appDirLegacy = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, LegacyConfigName);
            if (File.Exists(appDirLegacy)) return Load(appDirLegacy);

            // Fallback to default path (will create if LoadOrCreate is used)
            return LoadOrCreate(appDirDefault);
        }

        public static TTSConfig Load(string path)
        {
            _loadedPath = Path.GetFullPath(path);
            if (!File.Exists(_loadedPath)) return new TTSConfig();
            try
            {
                var content = File.ReadAllText(_loadedPath);
                var extension = Path.GetExtension(_loadedPath).ToLower();

                if (extension == ".yaml" || extension == ".yml")
                {
                    var deserializer = new DeserializerBuilder()
                        .WithNamingConvention(PascalCaseNamingConvention.Instance)
                        .IgnoreUnmatchedProperties()
                        .Build();
                    return deserializer.Deserialize<TTSConfig>(content) ?? new TTSConfig();
                }
                else
                {
                    // Fallback to JSON
                    var options = new System.Text.Json.JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true,
                        ReadCommentHandling = System.Text.Json.JsonCommentHandling.Skip
                    };
                    return System.Text.Json.JsonSerializer.Deserialize<TTSConfig>(content, options) ?? new TTSConfig();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading config from {path}: {ex.Message}, using defaults.");
                return new TTSConfig();
            }
        }

        public static TTSConfig LoadOrCreate(string path)
        {
            _loadedPath = path;
            if (!File.Exists(path))
            {
                Console.WriteLine($"Config file not found at '{path}'. Creating default config...");
                var defaultConfig = new TTSConfig();
                defaultConfig.Save(path);
                Console.WriteLine($"Default config created at '{path}'.");
                return defaultConfig;
            }
            return Load(path);
        }

        public static TTSConfig Reload()
        {
            if (string.IsNullOrEmpty(_loadedPath))
                return Load();
            return Load(_loadedPath);
        }

        public void Save(string path)
        {
            var extension = Path.GetExtension(path).ToLower();
            if (extension == ".yaml" || extension == ".yml")
            {
                var serializer = new SerializerBuilder()
                    .WithNamingConvention(PascalCaseNamingConvention.Instance)
                    .Build();
                var yaml = serializer.Serialize(this);
                File.WriteAllText(path, yaml);
            }
            else
            {
                var options = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
                var json = System.Text.Json.JsonSerializer.Serialize(this, options);
                File.WriteAllText(path, json);
            }
        }

        // ============================================================
        // 路径诊断输出
        // ============================================================

        public void PrintPaths()
        {
            Console.WriteLine("=== AstraTTS 路径配置 ===");
            Console.WriteLine($"ResourcesDir: {ResourcesDir}");
            Console.WriteLine($"SharedDir: {SharedDir}");
            Console.WriteLine($"DictionariesDir: {DictionariesDir}");
            Console.WriteLine($"ModelsV1Dir: {ModelsV1Dir}");
            Console.WriteLine($"ModelsV2Dir: {ModelsV2Dir}");
            Console.WriteLine($"AvatarsDir: {AvatarsDir}");
            Console.WriteLine($"UseEngineV2: {UseEngineV2}");
        }
    }
}
