using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;

namespace AstraTTS.Core.Core
{
    /// <summary>
    /// DirectML 设备端 KV 解码循环。
    ///
    /// 经典路径每生成一个 token 都要把 24 层 K/V（48 个张量）复制回托管内存，
    /// 下一轮再作为输入送回去；本类改为通过 ONNX Runtime 的 I/O Binding：
    ///   - present_k/v 输出直接落到 DML 显存，并作为下一步 past_k/v 输入直接使用；
    ///   - y_emb 同样常驻显存（它随序列增长，是第二大张量）；
    ///   - 仅 y（int64 token 序列）与 stop_condition_tensor（标量）回传 CPU，
    ///     用于取最后 token 与停止判断。
    ///
    /// 已验证（Windows + ORT 1.23.0 DirectML + AMD RX 5500 XT，真实模型）：
    ///   - first_stage_decoder 与 stage_decoder 是两个 Session，
    ///     跨 Session 消费设备张量必须在写入侧显式同步（Begin 中的 SynchronizeBoundOutputs）；
    ///   - 同一 Session 内设备张量自回喂按队列顺序执行，无需逐步同步；
    ///   - 动态增长的第一维（past_sequence_len 每步 +1）可正常绑定；
    ///   - 失败一律抛出异常，由调用方回退经典路径。
    /// </summary>
    public sealed class DmlKvDecodeLoop : IDisposable
    {
        private readonly InferenceSession _firstStage;
        private readonly InferenceSession _stage;
        private readonly bool _debug;
        private readonly RunOptions _runOptions = new RunOptions();
        private readonly OrtMemoryInfo _cpuMem = OrtMemoryInfo.DefaultInstance;
        private readonly OrtMemoryInfo _dmlMem;

        private readonly List<OrtValue> _kv = new List<OrtValue>(48);
        private IDisposableReadOnlyCollection<OrtValue>? _outputOwner;
        private OrtValue? _yEmbValue;
        private long[] _yBuffer = new long[512];   // 复用缓冲，避免每步分配/GC
        private long _yLength;

        // 名称缓存：避免每步字符串插值与按名线性查找。
        private readonly string[] _pastKNames;
        private readonly string[] _pastVNames;
        private readonly string[] _presentKNames;
        private readonly string[] _presentVNames;

        // 输出索引缓存：first_stage 与 stage_decoder 是两个模型、输出顺序不同，
        // 必须各自维护一套索引（首次 Run 后建立；同一 Session 的输出顺序稳定）。
        private int _fsYIdx = -1, _fsYEmbIdx = -1;
        private int[]? _fsPresentKIdx;
        private int[]? _fsPresentVIdx;
        private int _yIdx = -1, _yEmbIdx = -1, _stopIdx = -1;
        private int[]? _presentKIdx;
        private int[]? _presentVIdx;

        // 跨 Session（first stage → stage）交接后，首个 step 需要同步；
        // 同一 Session 内后续 step 由设备队列保序，无需逐步同步。
        private bool _needSync = true;
        private bool _stageMapBuilt;

        /// <summary>K/V 缓存层数（由 first_stage_decoder 的输出名推导）。</summary>
        public int NumLayers { get; }

        /// <summary>First Stage Decoder 输出的最后一个 token。</summary>
        public long FirstStageToken { get; private set; }

        /// <summary>已执行的 Stage Decoder 步数。</summary>
        public int StepIndex { get; private set; }

        public DmlKvDecodeLoop(InferenceSession firstStage, InferenceSession stageDecoder, int deviceId, bool debug)
        {
            _firstStage = firstStage ?? throw new ArgumentNullException(nameof(firstStage));
            _stage = stageDecoder ?? throw new ArgumentNullException(nameof(stageDecoder));
            _debug = debug;

            // 设备内存描述：与 DML EP 在会话内注册的分配器一致（名称 "DML"、设备分配器）。
            _dmlMem = new OrtMemoryInfo("DML", OrtAllocatorType.DeviceAllocator, deviceId, OrtMemType.Default);

            NumLayers = firstStage.OutputMetadata.Keys.Count(k => k.StartsWith("present_k_layer_", StringComparison.Ordinal));
            if (NumLayers <= 0)
                throw new InvalidOperationException("first_stage_decoder 未提供 present_k_layer_* 输出，无法使用设备 KV。");
            if (!stageDecoder.InputMetadata.ContainsKey("past_k_layer_0"))
                throw new InvalidOperationException("stage_decoder 缺少 past_k_layer_* 输入，无法使用设备 KV。");
            foreach (var required in new[] { "y", "y_emb", "stop_condition_tensor" })
            {
                if (!stageDecoder.OutputMetadata.ContainsKey(required))
                    throw new InvalidOperationException($"stage_decoder 缺少输出 '{required}'，无法使用设备 KV。");
            }

            _pastKNames = new string[NumLayers];
            _pastVNames = new string[NumLayers];
            _presentKNames = new string[NumLayers];
            _presentVNames = new string[NumLayers];
            for (int i = 0; i < NumLayers; i++)
            {
                _pastKNames[i] = $"past_k_layer_{i}";
                _pastVNames[i] = $"past_v_layer_{i}";
                _presentKNames[i] = $"present_k_layer_{i}";
                _presentVNames[i] = $"present_v_layer_{i}";
            }
        }

        /// <summary>
        /// 执行 First Stage Decoder 并把 K/V、y_emb 留在显存，初始化本轮解码。
        /// 调用方在结束本轮（无论成功与否）后必须调用 <see cref="EndAttempt"/>。
        /// </summary>
        /// <param name="x">T2S Encoder 输出 x 的扁平数据。</param>
        /// <param name="xShape">x 的完整形状（如 [1, xLength, 512]）。</param>
        /// <param name="prompts">T2S Encoder 输出 prompts。</param>
        public long Begin(float[] x, long[] xShape, long[] prompts)
        {
            EndAttempt();

            using var io = _firstStage.CreateIoBinding();
            using var xValue = OrtValue.CreateTensorValueFromMemory(x, xShape);
            using var promptsValue = OrtValue.CreateTensorValueFromMemory(prompts, new long[] { 1, prompts.Length });

            io.BindInput("x", xValue);
            io.BindInput("prompts", promptsValue);

            // 控制张量（token 序列）回传 CPU，便于读取最后 token。
            if (!_firstStage.OutputMetadata.ContainsKey("y"))
                throw new InvalidOperationException("first_stage_decoder 缺少输出 'y'。");
            io.BindOutputToDevice("y", _cpuMem);
            io.BindOutputToDevice("y_emb", _dmlMem);
            for (int layer = 0; layer < NumLayers; layer++)
            {
                io.BindOutputToDevice($"present_k_layer_{layer}", _dmlMem);
                io.BindOutputToDevice($"present_v_layer_{layer}", _dmlMem);
            }

            _firstStage.RunWithBinding(_runOptions, io);

            // first_stage_decoder 与 stage_decoder 是两个 Session，
            // 跨 Session 消费设备张量前必须显式同步（实测不同步会读到旧值）。
            io.SynchronizeBoundOutputs();

            var names = io.GetOutputNames();
            var outs = io.GetOutputValues();
            _outputOwner = outs; // 先接管所有权，保证异常路径也能释放

            BuildFirstStageIndexes(names);   // first stage 没有 stop 输出

            var yValue = outs[_fsYIdx];
            DetermineYLength(yValue, out _yLength);
            EnsureYBuffer(_yLength);
            yValue.GetTensorMutableDataAsSpan<long>().CopyTo(_yBuffer);
            _yEmbValue = outs[_fsYEmbIdx];

            _kv.Clear();
            for (int layer = 0; layer < NumLayers; layer++)
            {
                _kv.Add(outs[_fsPresentKIdx![layer]]);
                _kv.Add(outs[_fsPresentVIdx![layer]]);
            }

            StepIndex = 0;
            _needSync = true;   // first stage 与 stage 是不同 Session，交接后必须同步
            FirstStageToken = _yBuffer[_yLength - 1];
            if (_debug) Console.WriteLine($"[DirectML] 设备 KV 已初始化：{NumLayers} 层，first stage tokens={_yLength}");
            return FirstStageToken;
        }

        /// <summary>
        /// 执行一步 Stage Decoder。K/V 与 y_emb 在显存中直接流转，
        /// 只有新增 token 与停止标志回到 CPU。
        /// </summary>
        public void RunStep(out long lastToken, out bool stopCondition)
        {
            if (_yEmbValue == null || _kv.Count != NumLayers * 2)
                throw new InvalidOperationException("设备 KV 尚未初始化，请先调用 Begin。");

            using var io = _stage.CreateIoBinding();
            using var iy = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance, _yBuffer.AsMemory(0, (int)_yLength), new long[] { 1, _yLength });

            io.BindInput("iy", iy);
            io.BindInput("iy_emb", _yEmbValue);  // 常驻显存，跨步直接复用
            for (int layer = 0; layer < NumLayers; layer++)
            {
                io.BindInput(_pastKNames[layer], _kv[layer * 2]);
                io.BindInput(_pastVNames[layer], _kv[layer * 2 + 1]);
            }

            io.BindOutputToDevice("y", _cpuMem);
            io.BindOutputToDevice("y_emb", _dmlMem);
            io.BindOutputToDevice("stop_condition_tensor", _cpuMem);
            for (int layer = 0; layer < NumLayers; layer++)
            {
                io.BindOutputToDevice(_presentKNames[layer], _dmlMem);
                io.BindOutputToDevice(_presentVNames[layer], _dmlMem);
            }

            // 仅首个 step（消费上一个 Session 写入的设备张量）需要同步；
            // 同一 Session 内后续 step 按设备队列有序，实测数值精确。
            if (_needSync)
            {
                io.SynchronizeBoundInputs();
                _needSync = false;
            }

            _stage.RunWithBinding(_runOptions, io);

            var outs = io.GetOutputValues();
            if (!_stageMapBuilt)
            {
                BuildStageIndexes(io.GetOutputNames());
                _stageMapBuilt = true;
            }

            var yValue = outs[_yIdx];
            var stopValue = outs[_stopIdx];
            DetermineYLength(yValue, out long newLen);
            EnsureYBuffer(newLen);
            yValue.GetTensorMutableDataAsSpan<long>().CopyTo(_yBuffer);
            _yLength = newLen;

            lastToken = _yBuffer[newLen - 1];
            stopCondition = stopValue.GetTensorMutableDataAsSpan<bool>()[0];

            var newYEmb = outs[_yEmbIdx];
            var newKv = new List<OrtValue>(NumLayers * 2);
            for (int layer = 0; layer < NumLayers; layer++)
            {
                newKv.Add(outs[_presentKIdx![layer]]);
                newKv.Add(outs[_presentVIdx![layer]]);
            }

            // 上一轮输出（本轮输入）已随本轮 Run 消费完毕，可以释放。
            _outputOwner?.Dispose();
            _outputOwner = outs;

            _kv.Clear();
            _kv.AddRange(newKv);
            _yEmbValue = newYEmb;
            StepIndex++;
        }

        /// <summary>建立 First Stage 输出名 → 索引映射（无 stop 输出，仅一次）。</summary>
        private void BuildFirstStageIndexes(IReadOnlyList<string> names)
        {
            _fsPresentKIdx ??= new int[NumLayers];
            _fsPresentVIdx ??= new int[NumLayers];
            _fsYIdx = _fsYEmbIdx = -1;
            for (int i = 0; i < names.Count; i++)
            {
                string n = names[i];
                if (n == "y") { _fsYIdx = i; continue; }
                if (n == "y_emb") { _fsYEmbIdx = i; continue; }
                for (int layer = 0; layer < NumLayers; layer++)
                {
                    if (n == _presentKNames[layer]) { _fsPresentKIdx[layer] = i; break; }
                    if (n == _presentVNames[layer]) { _fsPresentVIdx[layer] = i; break; }
                }
            }
            if (_fsYIdx < 0 || _fsYEmbIdx < 0)
                throw new InvalidOperationException("first_stage_decoder 输出缺少 y / y_emb。");
        }

        /// <summary>建立 Stage Decoder 输出名 → 索引映射（仅一次）。</summary>
        private void BuildStageIndexes(IReadOnlyList<string> names)
        {
            _presentKIdx ??= new int[NumLayers];
            _presentVIdx ??= new int[NumLayers];
            _yIdx = _yEmbIdx = _stopIdx = -1;
            for (int i = 0; i < names.Count; i++)
            {
                string n = names[i];
                if (n == "y") { _yIdx = i; continue; }
                if (n == "y_emb") { _yEmbIdx = i; continue; }
                if (n == "stop_condition_tensor") { _stopIdx = i; continue; }
                for (int layer = 0; layer < NumLayers; layer++)
                {
                    if (n == _presentKNames[layer]) { _presentKIdx[layer] = i; break; }
                    if (n == _presentVNames[layer]) { _presentVIdx[layer] = i; break; }
                }
            }
            if (_yIdx < 0 || _yEmbIdx < 0 || _stopIdx < 0)
                throw new InvalidOperationException("stage_decoder 输出缺少 y / y_emb / stop_condition_tensor。");
        }

        /// <summary>取输出张量第一维长度（int64 序列长度）。</summary>
        private static void DetermineYLength(OrtValue yValue, out long length)
        {
            var dims = yValue.GetTensorTypeAndShape().Shape;
            length = dims[dims.Length - 1];
        }

        private void EnsureYBuffer(long length)
        {
            if (_yBuffer.Length < length)
            {
                _yBuffer = new long[Math.Max(length, _yBuffer.Length * 2)];
            }
        }

        /// <summary>
        /// 结束本轮尝试并释放设备张量；循环实例可继续用于下一次尝试（含重试）。
        /// </summary>
        public void EndAttempt()
        {
            _outputOwner?.Dispose();
            _outputOwner = null;
            _kv.Clear();
            _yEmbValue = null;
            _stageMapBuilt = false;
            _yLength = 0;
            StepIndex = 0;
            FirstStageToken = 0;
        }

        public void Dispose()
        {
            EndAttempt();
            _runOptions.Dispose();
        }
    }
}
