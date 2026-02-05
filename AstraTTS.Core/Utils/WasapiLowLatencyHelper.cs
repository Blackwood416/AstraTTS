using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using NAudio.CoreAudioApi;
using NAudio.Wave;

namespace AstraTTS.Core.Utils
{
    /// <summary>
    /// WASAPI 低延迟助手 (音频锚点锚定)
    /// 原理：通过创建一个极小周期的 IAudioClient3 流，强制 Windows 音频引擎进入低延迟调度模式 (通常从 10ms 降至 2.67ms)
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class WasapiLowLatencyHelper : IDisposable
    {
        private object? _audioClient;
        private bool _isStarted;

        // IAudioClient3 接口 ID
        private static readonly Guid IID_IAudioClient3 = new Guid("7ED4EE07-8E67-4CD4-8C1A-2B7A5987AD42");

        [ComImport]
        [Guid("7ED4EE07-8E67-4CD4-8C1A-2B7A5987AD42")]
        [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        internal interface IAudioClient3
        {
            [PreserveSig] int Initialize(int shareMode, int streamFlags, long hnsBufferDuration, long hnsPeriodicity, [In] WaveFormat pformat, [In] ref Guid audioSessionGuid);
            [PreserveSig] int GetBufferSize(out uint bufferSize);
            [PreserveSig] int GetStreamLatency(out long latency);
            [PreserveSig] int GetCurrentPadding(out uint currentPadding);
            [PreserveSig] int IsFormatSupported(int shareMode, [In] WaveFormat pFormat, out IntPtr ppClosestMatch);
            [PreserveSig] int GetMixFormat(out IntPtr ppDeviceFormat);
            [PreserveSig] int GetDevicePeriod(out long defaultPeriod, out long minimumPeriod);
            [PreserveSig] int Start();
            [PreserveSig] int Stop();
            [PreserveSig] int Reset();
            [PreserveSig] int SetEventHandle(IntPtr eventHandle);
            [PreserveSig] int GetService([In] ref Guid interfaceId, [MarshalAs(UnmanagedType.IUnknown)] out object interfacePointer);

            // IAudioClient3 specific methods
            [PreserveSig] int GetSharedModeEnginePeriod([In] WaveFormat pFormat, out uint pDefaultPeriodInFrames, out uint pFundamentalPeriodInFrames, out uint pMinPeriodInFrames, out uint pMaxPeriodInFrames);
            [PreserveSig] int GetCurrentSharedModeEnginePeriod(out IntPtr ppFormat, out uint pCurrentPeriodInFrames);
            [PreserveSig] int InitializeSharedAudioStream(uint StreamFlags, uint PeriodInFrames, [In] WaveFormat pFormat, [In] ref Guid AudioSessionGuid);
        }

        public void EnableLowLatency()
        {
            if (_isStarted) return;

            try
            {
                Console.WriteLine("[Wasapi] 正在尝试启用系统级低延迟模式 (Audio Anchor)...");

                // 1. 获取默认渲染设备
                using var enumerator = new MMDeviceEnumerator();
                using var device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);

                // 2. 激活 IAudioClient3
                // NAudio 的 MMDevice 不直接暴露 Activate。
                // 我们通过 device.AudioClient 获取 IAudioClient，然后 QueryInterface 获取 IAudioClient3
                var audioClient = device.AudioClient;
                var iid = IID_IAudioClient3;
                Marshal.QueryInterface(Marshal.GetIUnknownForObject(audioClient), ref iid, out var comPtr);

                if (comPtr == IntPtr.Zero)
                {
                    Console.WriteLine("   ⚠️ 无法获取 IAudioClient3 接口 (系统版本可能低于 Win10)");
                    return;
                }

                var client3 = (IAudioClient3)Marshal.GetObjectForIUnknown(comPtr);
                _audioClient = client3;

                // 3. 获取混合格式 (引擎原生格式)
                client3.GetMixFormat(out var formatPtr);
                var format = Marshal.PtrToStructure<WaveFormat>(formatPtr);

                // 4. 查询驱动支持的最小周期 (Period)
                client3.GetSharedModeEnginePeriod(format, out _, out _, out uint minPeriod, out _);
                Console.WriteLine($"   硬件支持最小周期: {minPeriod} frames ({(minPeriod * 1000.0 / format.SampleRate):F2} ms)");

                // 5. 初始化低延迟流
                // AUDCLNT_STREAMFLAGS_EVENTCALLBACK (0x00040000)
                uint flags = 0x00040000;
                Guid sessionGuid = Guid.Empty;
                int hr = client3.InitializeSharedAudioStream(flags, minPeriod, format, ref sessionGuid);

                if (hr < 0)
                {
                    Console.WriteLine($"   ❌ 初始化低延迟流失败: 0x{hr:X8}");
                    return;
                }

                // 必须关联一个 EventHandle 才能 Start
                var hEvent = CreateEvent(IntPtr.Zero, false, false, null);
                client3.SetEventHandle(hEvent);

                // 6. 启动锚点流
                client3.Start();
                _isStarted = true;

                Console.WriteLine("   ✅ 低延迟锚点已启动，Windows 音频引擎现已进入高性能模式。");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ⚠️ 启用低延迟失败: {ex.Message}");
            }
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr CreateEvent(IntPtr lpEventAttributes, bool bManualReset, bool bInitialState, string? lpName);

        public void Dispose()
        {
            if (_audioClient is IAudioClient3 client)
            {
                client.Stop();
                Marshal.ReleaseComObject(client);
            }
            _audioClient = null;
        }
    }
}
