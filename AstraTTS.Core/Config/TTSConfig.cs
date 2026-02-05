using System.Text.Json.Serialization;
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
        /// V1 引擎模型基础目录
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
        public string HubertPath => Path.Combine(ModelsV1Dir, "hubert", "chinese-hubert-base_full.onnx");

        [JsonIgnore]
        public string SpeakerEncoderPath => Path.Combine(ModelsV1Dir, "speaker_encoder.onnx");

        [JsonIgnore]
        public string V1TtsDir => Path.Combine(ModelsV1Dir, "tts");

        [JsonIgnore]
        public string BertModelPath => Path.Combine(ModelsV1Dir, "bert", "roberta.onnx");

        [JsonIgnore]
        public string TokenizerJsonPath => Path.Combine(ModelsV1Dir, "bert", "tokenizer", "tokenizer.json");

        // ============================================================
        // 按 Avatar 获取 V1 路径
        // ============================================================

        public string GetHubertPath(string avatarId) => Path.Combine(GetModelsV1Dir(avatarId), "hubert", "chinese-hubert-base_full.onnx");
        public string GetSpeakerEncoderPath(string avatarId) => Path.Combine(GetModelsV1Dir(avatarId), "speaker_encoder.onnx");
        public string GetV1TtsDir(string avatarId) => Path.Combine(GetModelsV1Dir(avatarId), "tts");
        public string GetBertModelPath(string avatarId) => Path.Combine(GetModelsV1Dir(avatarId), "bert", "roberta.onnx");
        public string GetTokenizerJsonPath(string avatarId) => Path.Combine(GetModelsV1Dir(avatarId), "bert", "tokenizer", "tokenizer.json");

        // ============================================================
        // 共享资源路径 (G2P 字典)
        // ============================================================

        [JsonIgnore]
        public string CmuDict => Path.Combine(DictionariesDir, "cmudict.dict");

        [JsonIgnore]
        public string PinyinDict => Path.Combine(DictionariesDir, "mandarin_pinyin.dict");

        [JsonIgnore]
        public string ChineseG2PDict => Path.Combine(DictionariesDir, "opencpop-strict.txt");

        [JsonIgnore]
        public string NeuralG2PModel => Path.Combine(SharedDir, "g2p", "checkpoint20.npz");

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
        public (string audioPath, string text)? GetDefaultReferenceAudio()
        {
            var avatar = GetDefaultAvatar();
            if (avatar == null) return null;

            var reference = avatar.GetDefaultReference();
            if (reference == null) return null;

            var fullPath = reference.GetFullAudioPath(AvatarsDir, avatar.Id);
            return (fullPath, reference.Text);
        }

        // ============================================================
        // 参考音频 (兼容旧版配置 - 如果没有 Avatar 配置则使用)
        // ============================================================

        public string? RefAudioPath { get; set; }
        public string RefText { get; set; } = "良宵方始，不必心急。";

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
        /// 内存优化级别 (0=禁用, 1=基础, 2=激进)。
        /// </summary>
        public int MemoryOptimizationLevel { get; set; } = 1;

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
        public int StreamingChunkTokens { get; set; } = 24;
        public bool StreamingMode { get; set; } = true;

        // ============================================================
        // 音频后端
        // ============================================================

        public bool WasapiExclusiveMode { get; set; } = true;
        public int LockFreeBufferSize { get; set; } = 65536 * 32;

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
            /// 是否启用数字转中文/英文规格化
            /// </summary>
            public bool EnableNormalization { get; set; } = true;
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
        public static string DefaultConfigPath => Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "config.json");

        public static TTSConfig Load()
        {
            return LoadOrCreate(DefaultConfigPath);
        }

        public static TTSConfig Load(string path)
        {
            _loadedPath = path;
            if (!File.Exists(path)) return new TTSConfig();
            try
            {
                var json = File.ReadAllText(path);
                var options = new System.Text.Json.JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    ReadCommentHandling = System.Text.Json.JsonCommentHandling.Skip
                };
                return System.Text.Json.JsonSerializer.Deserialize<TTSConfig>(json, options) ?? new TTSConfig();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading config: {ex.Message}, using defaults.");
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
            var options = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
            var json = System.Text.Json.JsonSerializer.Serialize(this, options);
            File.WriteAllText(path, json);
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
