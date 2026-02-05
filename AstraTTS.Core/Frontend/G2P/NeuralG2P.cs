using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using NumSharp;

namespace AstraTTS.Core.Frontend.G2P
{
    /// <summary>
    /// 基于神经网络的 G2P 转换器，用于处理 OOV (未登录) 词汇。
    /// 使用 GRU 编码器-解码器架构，从 checkpoint20.npz 加载预训练权重。
    /// </summary>
    public class NeuralG2P
    {
        // 字母表 (graphemes)
        private static readonly string[] Graphemes = {
            "<pad>", "<unk>", "</s>",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
        };

        // 音素表 (phonemes)
        private static readonly string[] Phonemes = {
            "<pad>", "<unk>", "<s>", "</s>",
            "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0",
            "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2",
            "B", "CH", "D", "DH",
            "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2",
            "F", "G", "HH",
            "IH0", "IH1", "IH2", "IY0", "IY1", "IY2",
            "JH", "K", "L", "M", "N", "NG",
            "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
            "P", "R", "S", "SH", "T", "TH",
            "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2",
            "V", "W", "Y", "Z", "ZH"
        };

        // 索引映射
        private readonly Dictionary<char, int> _g2idx;
        private readonly Dictionary<int, string> _idx2p;
        private readonly int _unkIdx;
        private readonly int _eosIdx;
        private readonly int _sosIdx;
        private readonly int _eosPhIdx;

        // 模型权重
        private NDArray? _encEmb;
        private NDArray? _encWIh, _encWHh, _encBIh, _encBHh;
        private NDArray? _decEmb;
        private NDArray? _decWIh, _decWHh, _decBIh, _decBHh;
        private NDArray? _fcW, _fcB;

        private bool _isLoaded = false;

        public NeuralG2P()
        {
            // 构建字母到索引映射
            _g2idx = new Dictionary<char, int>();
            for (int i = 0; i < Graphemes.Length; i++)
            {
                if (Graphemes[i].Length == 1)
                {
                    _g2idx[Graphemes[i][0]] = i;
                }
            }
            _unkIdx = Array.IndexOf(Graphemes, "<unk>");
            _eosIdx = Array.IndexOf(Graphemes, "</s>");

            // 构建索引到音素映射
            _idx2p = new Dictionary<int, string>();
            for (int i = 0; i < Phonemes.Length; i++)
            {
                _idx2p[i] = Phonemes[i];
            }
            _sosIdx = Array.IndexOf(Phonemes, "<s>");
            _eosPhIdx = Array.IndexOf(Phonemes, "</s>");
        }

        /// <summary>
        /// 从 NPZ 文件加载模型权重。
        /// 手动解析 NPZ (ZIP with NPY files)
        /// </summary>
        public void LoadModel(string npzPath)
        {
            if (!File.Exists(npzPath))
            {
                Console.WriteLine($"[NeuralG2P] Model file not found: {npzPath}");
                return;
            }

            try
            {
                Console.WriteLine($"[NeuralG2P] Loading model from {npzPath}...");
                
                using (var archive = ZipFile.OpenRead(npzPath))
                {
                    _encEmb = LoadNpy(archive, "enc_emb.npy");
                    _encWIh = LoadNpy(archive, "enc_w_ih.npy");
                    _encWHh = LoadNpy(archive, "enc_w_hh.npy");
                    _encBIh = LoadNpy(archive, "enc_b_ih.npy");
                    _encBHh = LoadNpy(archive, "enc_b_hh.npy");
                    
                    _decEmb = LoadNpy(archive, "dec_emb.npy");
                    _decWIh = LoadNpy(archive, "dec_w_ih.npy");
                    _decWHh = LoadNpy(archive, "dec_w_hh.npy");
                    _decBIh = LoadNpy(archive, "dec_b_ih.npy");
                    _decBHh = LoadNpy(archive, "dec_b_hh.npy");
                    
                    _fcW = LoadNpy(archive, "fc_w.npy");
                    _fcB = LoadNpy(archive, "fc_b.npy");
                }

                _isLoaded = true;
                if (!object.ReferenceEquals(_encEmb, null))
                {
                    Console.WriteLine($"[NeuralG2P] Model loaded successfully. Encoder embedding shape: {string.Join(",", _encEmb.shape)}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[NeuralG2P] Failed to load model: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
                _isLoaded = false;
            }
        }

        private NDArray? LoadNpy(ZipArchive archive, string entryName)
        {
            var entry = archive.GetEntry(entryName);
            if (entry == null) 
            {
                Console.WriteLine($"[NeuralG2P] Warning: Entry {entryName} not found in NPZ.");
                return null;
            }

            using (var stream = entry.Open())
            using (var ms = new MemoryStream())
            {
                stream.CopyTo(ms);
                ms.Position = 0;
                
                // NPY Header Parsing
                // 1. Magic string (6 bytes): \x93NUMPY
                byte[] magic = new byte[6];
                ms.Read(magic, 0, 6);
                
                // 2. Version (2 bytes)
                byte[] version = new byte[2];
                ms.Read(version, 0, 2);
                
                // 3. Header len (2 bytes, Little Endian)
                byte[] headerLenBytes = new byte[2];
                ms.Read(headerLenBytes, 0, 2);
                int headerLen = BitConverter.ToUInt16(headerLenBytes, 0);
                
                // 4. Header dict string
                byte[] headerBytes = new byte[headerLen];
                ms.Read(headerBytes, 0, headerLen);
                string headerStr = Encoding.ASCII.GetString(headerBytes).Trim();
                
                // Parse Shape and Dtype
                // example: {'descr': '<f4', 'fortran_order': False, 'shape': (29, 256), }
                var shapeMatch = Regex.Match(headerStr, @"'shape': \((.*?)\)");
                if (!shapeMatch.Success)
                {
                    Console.WriteLine($"[NeuralG2P] Error: Failed to parse shape from header: {headerStr}");
                    return null;
                }
                var shapeStrs = shapeMatch.Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries);
                var shape = shapeStrs.Select(s => int.Parse(s.Trim())).ToArray();
                
                // Read Data
                int totalElements = shape.Aggregate(1, (a, b) => a * b);
                int bytesToRead = totalElements * 4; // float32 = 4 bytes
                
                byte[] dataBytes = new byte[bytesToRead];
                ms.Read(dataBytes, 0, bytesToRead);
                
                float[] data = new float[totalElements];
                Buffer.BlockCopy(dataBytes, 0, data, 0, bytesToRead);
                
                var nd = np.array(data);
                return nd.reshape(shape);
            }
        }

        /// <summary>
        /// 预测单词的音素序列。
        /// </summary>
        public List<string> Predict(string word)
        {
            if (!_isLoaded) return new List<string>();
            if (_encEmb is null) return new List<string>();

            try
            {
                // 1. 编码输入
                var enc = Encode(word);
                
                // 2. 运行 GRU 编码器
                int hiddenSize = _encWHh!.shape[1];
                var h = np.zeros(1, hiddenSize);
                
                for (int t = 0; t < enc.shape[1]; t++)
                {
                    var xt = enc[$":, {t}, :"];
                    h = GRUCell(xt, h, _encWIh!, _encWHh!, _encBIh!, _encBHh!);
                }
                
                // 3. 解码
                var preds = new List<int>();
                
                // dec = dec_emb[sos_idx]
                // Note: NumSharp slicing syntax is tricky, explicit buffer copy is safer
                var decEmbData = _decEmb!.GetData<float>().ToArray();
                int embDim = _decEmb.shape[1];
                float[] startEmb = new float[embDim];
                Array.Copy(decEmbData, _sosIdx * embDim, startEmb, 0, embDim);
                
                var dec = np.array(startEmb).reshape(1, embDim);
                
                for (int step = 0; step < 20; step++)  // Max 20 steps
                {
                    h = GRUCell(dec, h, _decWIh!, _decWHh!, _decBIh!, _decBHh!);
                    
                    // 线性层
                    var logits = np.matmul(h, _fcW!.T) + _fcB!;
                    
                    // Argmax
                    var logitsData = logits.GetData<float>().ToArray();
                    int predIdx = 0;
                    float maxVal = float.MinValue;
                    for(int i=0; i<logitsData.Length; i++) {
                        if(logitsData[i] > maxVal) {
                            maxVal = logitsData[i];
                            predIdx = i;
                        }
                    }
                    
                    if (predIdx == _eosPhIdx) break;
                    preds.Add(predIdx);
                    
                    // 下一步输入
                    Array.Copy(decEmbData, predIdx * embDim, startEmb, 0, embDim);
                    dec = np.array(startEmb).reshape(1, embDim);
                }
                
                // 4. 转换为音素
                return preds.Select(idx => _idx2p.GetValueOrDefault(idx, "<unk>")).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[NeuralG2P] Prediction error: {ex.Message}");
                return new List<string>();
            }
        }

        /// <summary>
        /// 将单词编码为嵌入序列。
        /// </summary>
        private NDArray Encode(string word)
        {
            var chars = word.ToLower().ToCharArray().ToList();
            var indices = new List<int>();
            
            foreach (var c in chars)
            {
                indices.Add(_g2idx.GetValueOrDefault(c, _unkIdx));
            }
            indices.Add(_eosIdx);  // 添加 </s>
            
            // 查找嵌入
            int seqLen = indices.Count;
            int embDim = _encEmb!.shape[1];
            
            // Manual lookup
            var embData = _encEmb.GetData<float>().ToArray();
            var resultData = new float[seqLen * embDim];
            
            for (int i = 0; i < seqLen; i++)
            {
                int idx = indices[i];
                Array.Copy(embData, idx * embDim, resultData, i * embDim, embDim);
            }
            
            return np.array(resultData).reshape(1, seqLen, embDim);
        }

        /// <summary>
        /// GRU Cell 前向传播。
        /// </summary>
        private NDArray GRUCell(NDArray x, NDArray h, NDArray wIh, NDArray wHh, NDArray bIh, NDArray bHh)
        {
            // rzn_ih = x @ w_ih.T + b_ih
            var rznIh = np.matmul(x, wIh.T) + bIh;
            // rzn_hh = h @ w_hh.T + b_hh
            var rznHh = np.matmul(h, wHh.T) + bHh;
            
            int hiddenSize = wHh.shape[1];
            
            var rznIhData = rznIh.GetData<float>().ToArray();
            var rznHhData = rznHh.GetData<float>().ToArray();
            
            float[] rData = new float[hiddenSize];
            float[] zData = new float[hiddenSize];
            float[] nData = new float[hiddenSize];
            
            for (int i = 0; i < hiddenSize; i++)
            {
                float rInput = rznIhData[i] + rznHhData[i];
                float zInput = rznIhData[i + hiddenSize] + rznHhData[i + hiddenSize];
                
                rData[i] = Sigmoid(rInput);
                zData[i] = Sigmoid(zInput);
            }
            
            for (int i = 0; i < hiddenSize; i++)
            {
                float nInput = rznIhData[i + 2 * hiddenSize] + rData[i] * rznHhData[i + 2 * hiddenSize];
                nData[i] = (float)Math.Tanh(nInput);
            }
            
            float[] hNewData = new float[hiddenSize];
            var hData = h.GetData<float>().ToArray();
            
            for (int i = 0; i < hiddenSize; i++)
            {
                hNewData[i] = (1 - zData[i]) * nData[i] + zData[i] * hData[i];
            }
            
            return np.array(hNewData).reshape(1, hiddenSize);
        }

        private float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }
        
        // NDArray Sigmoid implementation removed, using element-wise loop in GRUCell

        /// <summary>
        /// 测试神经网络 G2P。
        /// </summary>
        public static void Test(string modelPath)
        {
            var neuralG2p = new NeuralG2P();
            neuralG2p.LoadModel(modelPath);
            
            var testWords = new[] { "hello", "world", "python", "feature", "neural", "A", "E" };
            
            foreach (var word in testWords)
            {
                var phonemes = neuralG2p.Predict(word);
                Console.WriteLine($"[NeuralG2P Test] {word} -> {string.Join(" ", phonemes)}");
            }
        }
    }
}
