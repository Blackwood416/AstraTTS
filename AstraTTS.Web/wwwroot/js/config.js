// ============================================================
// AstraTTS WebUI — Config Module
// 控制中心表单渲染与保存、补全缺失参数 (13 个)
// ============================================================

const Config = {
    init() {
        document.getElementById('save-config-btn').addEventListener('click', () => this.save());
        const resetBtn = document.getElementById('reset-config-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.reset());
        }
        
        // Language enforcement (max-2)
        const container = document.getElementById('cfg-langs');
        if (container) {
            container.addEventListener('change', (e) => {
                if (e.target.type === 'checkbox') {
                    const checked = container.querySelectorAll('input[type="checkbox"]:checked');
                    if (checked.length > 2) {
                        App.showToast('目前最多支持同时选择两种语言', 'warn');
                        e.target.checked = false;
                    }
                }
            });
        }

        // Speed display update
        const form = document.getElementById('config-form');
        const speedInput = form.elements['Speed'];
        const speedVal = document.getElementById('cfg-speed-val');
        if (speedInput && speedVal) {
            speedInput.addEventListener('input', (e) => {
                speedVal.textContent = parseFloat(e.target.value).toFixed(1);
            });
        }
    },

    renderForm() {
        if (!App.state.config) return;
        const form = document.getElementById('config-form');
        const c = App.state.config;

        // --- Core ---
        form.ResourcesDir.value = App.getProp(c, 'resourcesDir') || '';
        form.UseEngineV2.checked = !!App.getProp(c, 'useEngineV2');
        form.GraphOptimizationLevel.value = App.getProp(c, 'graphOptimizationLevel') ?? 1;
        form.IntraOpNumThreads.value = App.getProp(c, 'intraOpNumThreads') ?? 0;
        form.InterOpNumThreads.value = App.getProp(c, 'interOpNumThreads') ?? 0;
        
        // --- Pool (NEW) ---
        if (form.PoolCapacity) form.PoolCapacity.value = App.getProp(c, 'poolCapacity') ?? 1;
        if (form.ReuseMemory) form.ReuseMemory.checked = App.getProp(c, 'reuseMemory') ?? true;

        // --- Synthesis Defaults ---
        if (form.Speed) form.Speed.value = App.getProp(c, 'speed') ?? 1.0;
        if (form.NoiseScale) form.NoiseScale.value = App.getProp(c, 'noiseScale') ?? 0.35;
        if (form.TopK) form.TopK.value = App.getProp(c, 'topK') ?? 15;
        if (form.Temperature) form.Temperature.value = App.getProp(c, 'temperature') ?? 1.0;
        if (form.StreamingChunkSize) form.StreamingChunkSize.value = App.getProp(c, 'streamingChunkSize') ?? 22;
        if (form.StreamingPreBufferChunks) form.StreamingPreBufferChunks.value = App.getProp(c, 'streamingPreBufferChunks') ?? 2;
        if (document.getElementById('cfg-speed-val')) {
             document.getElementById('cfg-speed-val').textContent = (App.getProp(c, 'speed') ?? 1.0).toFixed(1);
        }

        // --- Inference Retries (NEW) ---
        const inf = App.getProp(c, 'inference') || {};
        if (form['Inference.MaxRetries']) form['Inference.MaxRetries'].value = App.getProp(inf, 'maxRetries') ?? 3;
        if (form['Inference.MinTokenMultiplierChinese']) form['Inference.MinTokenMultiplierChinese'].value = App.getProp(inf, 'minTokenMultiplierChinese') ?? 2.1;
        if (form['Inference.MinTokenMultiplierJapanese']) form['Inference.MinTokenMultiplierJapanese'].value = App.getProp(inf, 'minTokenMultiplierJapanese') ?? 2.0;
        if (form['Inference.MinTokenMultiplierEnglish']) form['Inference.MinTokenMultiplierEnglish'].value = App.getProp(inf, 'minTokenMultiplierEnglish') ?? 1.3;
        if (form['Inference.MinTokenMultiplierOther']) form['Inference.MinTokenMultiplierOther'].value = App.getProp(inf, 'minTokenMultiplierOther') ?? 0.5;

        // --- Debug (NEW) ---
        if (form.DebugMode) form.DebugMode.checked = !!App.getProp(c, 'debugMode');

        // --- G2P ---
        const g2p = App.getProp(c, 'g2P') || App.getProp(c, 'g2p') || {};
        const allowedLangs = App.getProp(g2p, 'languages') || ['zh', 'en'];
        form.querySelectorAll('input[name="G2P.Languages"]').forEach(cb => {
            cb.checked = allowedLangs.includes(cb.value);
        });
        form['G2P.CustomDictPath'].value = App.getProp(g2p, 'customDictPath') || '';
        if (form['G2P.PriorityMode']) form['G2P.PriorityMode'].value = App.getProp(g2p, 'priorityMode') ?? 0;

        // --- Default Avatar ---
        const defaultId = App.getProp(c, 'defaultAvatarId') || '';
        const select = form.DefaultAvatarId;
        if (App.state.avatars.length === 0) {
            select.innerHTML = `<option value="${defaultId}">${defaultId || '(无可用音色)'}</option>`;
        } else {
            select.innerHTML = App.state.avatars.map(a => {
                const aid = App.getProp(a, 'id');
                const aname = App.getProp(a, 'name');
                return `<option value="${aid}"${aid === defaultId ? ' selected' : ''}>${aname || aid} (${aid})</option>`;
            }).join('');
        }
    },

    async save() {
        const form = document.getElementById('config-form');
        const cfg = { ...App.state.config };

        cfg.ResourcesDir = form.ResourcesDir.value;
        cfg.DefaultAvatarId = form.DefaultAvatarId.value;
        cfg.UseEngineV2 = form.UseEngineV2.checked;
        cfg.GraphOptimizationLevel = parseInt(form.GraphOptimizationLevel.value);
        cfg.IntraOpNumThreads = parseInt(form.IntraOpNumThreads.value);
        cfg.InterOpNumThreads = parseInt(form.InterOpNumThreads.value);
        
        // New core params
        cfg.PoolCapacity = parseInt(form.PoolCapacity?.value || 1);
        cfg.ReuseMemory = form.ReuseMemory?.checked ?? true;
        cfg.DebugMode = form.DebugMode?.checked ?? false;

        // Synthesis defaults
        cfg.Speed = parseFloat(form.Speed?.value || 1.0);
        cfg.NoiseScale = parseFloat(form.NoiseScale?.value || 0.35);
        cfg.TopK = parseInt(form.TopK?.value || 15);
        cfg.Temperature = parseFloat(form.Temperature?.value || 1.0);
        cfg.StreamingChunkSize = parseInt(form.StreamingChunkSize?.value || 22);
        cfg.StreamingPreBufferChunks = parseInt(form.StreamingPreBufferChunks?.value || 2);

        // Inference
        cfg.Inference = {
            MaxRetries: parseInt(form['Inference.MaxRetries']?.value || 3),
            MinTokenMultiplierChinese: parseFloat(form['Inference.MinTokenMultiplierChinese']?.value || 2.1),
            MinTokenMultiplierJapanese: parseFloat(form['Inference.MinTokenMultiplierJapanese']?.value || 2.0),
            MinTokenMultiplierEnglish: parseFloat(form['Inference.MinTokenMultiplierEnglish']?.value || 1.3),
            MinTokenMultiplierOther: parseFloat(form['Inference.MinTokenMultiplierOther']?.value || 0.5)
        };

        const selectedLangs = Array.from(form.querySelectorAll('input[name="G2P.Languages"]:checked')).map(cb => cb.value);
        cfg.G2P = {
            Languages: selectedLangs,
            CustomDictPath: form['G2P.CustomDictPath'].value,
            PriorityMode: parseInt(form['G2P.PriorityMode']?.value || 0)
        };

        App.showLoading('正在保存全局配置并重载引擎...');
        try {
            const resp = await fetch('/api/tts/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cfg)
            });
            const result = await resp.json();
            if (resp.ok && result.success) {
                App.showToast('全局配置已更新');
                await App.fetchAllWithRetry();
            } else {
                App.showToast('保存失败: ' + (result.message || '未知错误'), 'error');
            }
        } catch (e) {
            App.showToast('更新失败: ' + e.message, 'error');
        } finally {
            App.hideLoading();
        }
    },

    async reset() {
        if (!confirm('此操作将还原所有已配置项并清除音色库，是否确定？')) {
            return;
        }

        App.showLoading('正在重置全局配置并重载引擎...');
        try {
            const resp = await fetch('/api/tts/config/reset', {
                method: 'POST'
            });
            const result = await resp.json();
            if (resp.ok && result.success) {
                App.showToast('配置已重置为系统默认值');
                await App.fetchAllWithRetry();
            } else {
                App.showToast('重置失败: ' + (result.message || '未知错误'), 'error');
            }
        } catch (e) {
            App.showToast('重置失败: ' + e.message, 'error');
        } finally {
            App.hideLoading();
        }
    }
};
