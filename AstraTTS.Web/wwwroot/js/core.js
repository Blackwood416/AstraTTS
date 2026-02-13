// ============================================================
// AstraTTS WebUI — Core Module
// 全局状态管理、工具函数、导航、主题、Toast
// ============================================================

const App = {
    state: {
        config: null,
        info: null,
        status: null,
        avatars: [],
        activeTab: 'dashboard',
        isDark: true,
        currentEditingAvatar: null,
        lastConverterRefsDir: null
    },

    // Case-insensitive property accessor
    getProp(obj, propName) {
        if (!obj || !propName) return undefined;
        if (obj[propName] !== undefined) return obj[propName];
        const lower = propName.toLowerCase();
        const upper = propName.charAt(0).toUpperCase() + propName.slice(1);
        if (obj[lower] !== undefined) return obj[lower];
        if (obj[upper] !== undefined) return obj[upper];
        const found = Object.keys(obj).find(k => k.toLowerCase() === lower);
        return found ? obj[found] : undefined;
    },

    showLoading(text = '正在处理，请稍候...') {
        document.getElementById('loading-overlay').style.display = 'flex';
        document.getElementById('loading-text').textContent = text;
    },

    hideLoading() {
        document.getElementById('loading-overlay').style.display = 'none';
    },

    showToast(msg, type = 'success') {
        const toast = document.getElementById('toast');
        toast.textContent = msg;
        toast.style.background = type === 'error' ? 'linear-gradient(135deg, #ef4444, #b91c1c)' :
                                 (type === 'info' ? 'linear-gradient(135deg, #3b82f6, #1d4ed8)' : '#1e293b');
        toast.style.display = 'block';
        setTimeout(() => toast.style.display = 'none', 3000);
    },

    // --- Data Fetching ---

    async fetchAll() {
        console.log('[App] Fetching all data...');
        try {
            const [infoResp, configResp, statusResp] = await Promise.all([
                fetch('/api/tts/info', { cache: 'no-cache' }),
                fetch('/api/tts/config', { cache: 'no-cache' }),
                fetch('/api/tts/status', { cache: 'no-cache' }).catch(() => null)
            ]);

            if (!infoResp.ok || !configResp.ok) throw new Error('API request failed');

            this.state.info = await infoResp.json();
            this.state.config = await configResp.json();
            this.state.status = statusResp && statusResp.ok ? await statusResp.json() : null;
            this.state.avatars = this.state.config.avatars || this.state.config.Avatars || [];

            console.log('[App] Data loaded:', { info: this.state.info, avatarsCount: this.state.avatars.length });

            // Dispatch to all modules
            if (typeof Dashboard !== 'undefined') Dashboard.update();
            if (typeof Avatars !== 'undefined') Avatars.renderList();
            if (typeof Config !== 'undefined') Config.renderForm();
            if (typeof Playground !== 'undefined') Playground.updateAvatars();
        } catch (e) {
            console.error('[App] fetchAll Error:', e);
            this.state.status = null;
            if (typeof Dashboard !== 'undefined') Dashboard.updateStatus();
            this.showToast('数据同步失败，请检查控制台获取详情', 'error');
        }
    },

    async fetchAllWithRetry(maxRetries = 5, delayMs = 2000) {
        for (let i = 0; i < maxRetries; i++) {
            try {
                await this.fetchAll();
                return;
            } catch (e) {
                if (i < maxRetries - 1) {
                    await new Promise(r => setTimeout(r, delayMs));
                }
            }
        }
    },

    // --- Navigation ---

    setupNavigation() {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const tabId = item.dataset.tab;
                this.switchTab(tabId);
            });
        });
    },

    switchTab(tabId) {
        document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.tab === tabId));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.toggle('active', t.id === tabId + '-tab'));
        this.state.activeTab = tabId;
    },

    // --- Theme ---

    setupTheme() {
        const isDark = localStorage.getItem('theme') !== 'light';
        this.toggleTheme(isDark);
    },

    toggleTheme(dark) {
        this.state.isDark = dark;
        document.body.classList.toggle('dark-theme', dark);
        document.body.classList.toggle('light-theme', !dark);
        document.querySelector('.sun-icon').style.display = dark ? 'block' : 'none';
        document.querySelector('.moon-icon').style.display = dark ? 'none' : 'block';
        localStorage.setItem('theme', dark ? 'dark' : 'light');
    },

    // --- Status Polling ---

    async pollStatus() {
        try {
            const resp = await fetch('/api/tts/status', { cache: 'no-cache' });
            if (resp.ok) {
                this.state.status = await resp.json();
            } else {
                this.state.status = null;
            }
        } catch {
            this.state.status = null;
        }
        if (typeof Dashboard !== 'undefined') Dashboard.updateStatus();
    },

    // --- Init ---

    init() {
        this.setupNavigation();
        this.setupTheme();

        document.getElementById('theme-toggle').addEventListener('click', () => this.toggleTheme(!this.state.isDark));

        document.getElementById('reload-btn').addEventListener('click', async () => {
            const btn = document.getElementById('reload-btn');
            btn.disabled = true;
            this.showLoading('正在重载配置，引擎将重新初始化...');
            try {
                const resp = await fetch('/api/tts/reload', { method: 'POST' });
                if (resp.ok) {
                    this.showToast('引擎重载成功');
                    await this.fetchAllWithRetry();
                }
            } finally { btn.disabled = false; this.hideLoading(); }
        });

        this.fetchAll();
        setInterval(() => this.pollStatus(), 5000);
    }
};

document.addEventListener('DOMContentLoaded', () => {
    App.init();
    if (typeof Avatars !== 'undefined') Avatars.init();
    if (typeof Config !== 'undefined') Config.init();
    if (typeof Playground !== 'undefined') Playground.init();
    if (typeof Converter !== 'undefined') Converter.init();
});
