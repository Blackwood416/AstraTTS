using System.Text.Json;

namespace AstraTTS.Core.Config
{
    /// <summary>
    /// 模型配置 (对应 config.json)
    /// </summary>
    public class V2ModelConfig
    {
        public DataConfig? Data { get; set; }
        public ModelInfo? Model { get; set; }
        public ExportOptions? ExportOptions { get; set; }
        public Dictionary<string, int>? SymbolToId { get; set; }

        public static V2ModelConfig Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"模型配置文件不存在: {path}");

            var json = File.ReadAllText(path);
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            };

            return JsonSerializer.Deserialize<V2ModelConfig>(json, options)
                   ?? throw new InvalidDataException("无法解析模型配置文件");
        }
    }

    /// <summary>
    /// 音频数据配置
    /// </summary>
    public class DataConfig
    {
        public int SamplingRate { get; set; } = 32000;
        public int FilterLength { get; set; } = 2048;
        public int HopLength { get; set; } = 640;
        public int WinLength { get; set; } = 2048;
        public int NSpeakers { get; set; } = 300;
    }

    /// <summary>
    /// 模型信息
    /// </summary>
    public class ModelInfo
    {
        public string Version { get; set; } = "v2";
        public string SemanticFrameRate { get; set; } = "25hz";
    }

    /// <summary>
    /// 导出选项
    /// </summary>
    public class ExportOptions
    {
        public int MaxLen { get; set; } = 1000;
        public int OpsetVersion { get; set; } = 17;
        public string Target { get; set; } = "directml";
    }
}
