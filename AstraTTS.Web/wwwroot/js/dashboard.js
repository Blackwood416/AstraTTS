// ============================================================
// AstraTTS WebUI — Dashboard Module
// 仪表盘渲染、运行状态更新
// ============================================================

const Dashboard = {
    updateStatus() {
        const dot = document.querySelector('.status-dot');
        const label = document.getElementById('runtime-status');
        const statusVal = App.state.status ? (App.getProp(App.state.status, 'status') || '') : '';
        const isOnline = statusVal.toLowerCase() === 'online';
        const engineName = App.state.info ? (App.getProp(App.state.info, 'engine') || '') : '';

        dot.classList.toggle('online', isOnline);
        if (isOnline) {
            label.textContent = engineName ? engineName + ' — 引擎在线' : '引擎在线';
        } else {
            label.textContent = '引擎离线';
        }
    },

    update() {
        this.updateStatus();
        if (!App.state.info) return;
        document.getElementById('stat-sampling-rate').textContent = App.getProp(App.state.info, 'samplingRate') || '--';
        document.getElementById('stat-engine-type').textContent = App.getProp(App.state.info, 'engine') || '--';
        document.getElementById('stat-avatar-count').textContent = App.state.avatars.length;
        document.getElementById('stat-default-id').textContent = App.getProp(App.state.config, 'defaultAvatarId') || '--';
        document.getElementById('stat-resources-dir').textContent = App.getProp(App.state.config, 'resourcesDir') || '--';
        document.getElementById('stat-v2-enabled').textContent = App.getProp(App.state.config, 'useEngineV2') ? '已开启' : '未开启 (V1)';
    }
};
