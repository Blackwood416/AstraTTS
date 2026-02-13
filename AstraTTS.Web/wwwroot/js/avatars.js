// ============================================================
// AstraTTS WebUI — Avatars Module
// 音色库管理、编辑器、参考音频字段补充 (Name/Language)
// ============================================================

const Avatars = {
    init() {
        document.getElementById('add-avatar-btn').addEventListener('click', () => this.openEditor());
        document.getElementById('cancel-avatar-btn').addEventListener('click', () => {
            document.getElementById('avatar-editor').style.display = 'none';
        });

        document.getElementById('add-reference-btn').addEventListener('click', () => {
            this.syncFromDOM();
            App.state.currentEditingAvatar.references.push({
                id: 'ref_' + Date.now(),
                name: '',
                language: '',
                audioPath: '',
                text: ''
            });
            this.renderReferences();
            this.updateDefaultRefDropdown();
        });

        document.getElementById('import-audio-btn').addEventListener('click', () => this.handleImportAudio());
        document.getElementById('save-avatar-btn').addEventListener('click', () => this.save());
    },

    renderList() {
        const body = document.getElementById('avatar-list-body');
        if (!Array.isArray(App.state.avatars)) {
            body.innerHTML = '<tr><td colspan="5">音色列表格式错误</td></tr>';
            return;
        }
        body.innerHTML = App.state.avatars.map(a => {
            if (!a) return '';
            const id = App.getProp(a, 'id');
            const name = App.getProp(a, 'name');
            const useV2 = App.getProp(a, 'useEngineV2');
            const refs = App.getProp(a, 'references') || [];

            const engineLabel = useV2 === true ? 'V2' : (useV2 === false ? 'V1' : 'AUTO');
            return `
            <tr>
                <td><code>${id}</code></td>
                <td><strong>${name || 'Unnamed'}</strong></td>
                <td><span class="badge">${engineLabel}</span></td>
                <td>${refs.length} 个</td>
                <td>
                    <button class="text-btn edit-btn" data-id="${id}">编辑</button>
                    <button class="danger-btn delete-btn" data-id="${id}">删除</button>
                </td>
            </tr>
        `}).join('');

        body.querySelectorAll('.edit-btn').forEach(btn => {
            btn.addEventListener('click', () => this.openEditor(btn.dataset.id));
        });
        body.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', () => this.delete(btn.dataset.id));
        });
    },

    async openEditor(avatarId = null) {
        const isNew = !avatarId;
        const editor = document.getElementById('avatar-editor');
        if (avatarId) {
            const avatar = App.state.avatars.find(a => App.getProp(a, 'id') === avatarId);
            if (!avatar) {
                App.showToast('找不到音色: ' + avatarId, 'error');
                return;
            }
            App.state.currentEditingAvatar = JSON.parse(JSON.stringify(avatar));
            document.getElementById('editor-title').textContent = '编辑音色: ' + (App.getProp(avatar, 'name') || avatarId);
            document.getElementById('edit-avatar-is-new').value = 'false';
        } else {
            App.state.currentEditingAvatar = { id: '', name: '', description: '', useEngineV2: null, defaultReferenceId: 'default', references: [] };
            document.getElementById('editor-title').textContent = '新增音色配置';
            document.getElementById('edit-avatar-is-new').value = 'true';
        }

        const form = document.getElementById('avatar-form');
        form.elements.Name.value = App.getProp(App.state.currentEditingAvatar, 'name') || '';
        form.elements.Description.value = App.getProp(App.state.currentEditingAvatar, 'description') || '';
        form.elements.UseEngineV2.value = String(App.getProp(App.state.currentEditingAvatar, 'useEngineV2'));

        await this.loadIdOptions(App.getProp(App.state.currentEditingAvatar, 'id'), isNew);

        // Events for ID and Engine change
        const engineSelect = form.elements.UseEngineV2;
        const onEngineChange = () => {
            this.loadIdOptions(document.getElementById('edit-avatar-id').value, isNew);
        };
        engineSelect.onchange = onEngineChange;

        const idSelect = document.getElementById('edit-avatar-id');
        idSelect.onchange = () => this.renderReferences();

        this.renderReferences();
        this.updateDefaultRefDropdown();

        editor.style.display = 'block';
        editor.scrollIntoView({ behavior: 'smooth' });
    },

    async loadIdOptions(currentId, isNew) {
        const idSelect = document.getElementById('edit-avatar-id');
        const engineVal = document.getElementById('avatar-form').elements.UseEngineV2.value;
        let useV2 = engineVal === 'null' ? !!App.getProp(App.state.config, 'useEngineV2') : engineVal === 'true';

        try {
            const resp = await fetch(`/api/tts/fs/avatars?useV2=${useV2}`, { cache: 'no-cache' });
            const entries = await resp.json();
            const registeredIds = new Set(App.state.avatars.map(a => App.getProp(a, 'id')));

            let options = isNew ? '<option value="">— 请选择 —</option>' : '';
            entries.forEach(e => {
                if (isNew && registeredIds.has(e.id) && e.id !== currentId) return;
                let label = e.id + (e.valid ? ' ✓' : (!e.hasModel ? ' ⚠ 缺少模型' : ' ⚠ 缺少音色目录'));
                options += `<option value="${e.id}"${e.id === currentId ? ' selected' : ''}>${label}</option>`;
            });

            if (!isNew && currentId && !options.includes(`value="${currentId}"`)) {
                options = `<option value="${currentId}" selected>${currentId}</option>` + options;
            }
            idSelect.innerHTML = options;
            idSelect.disabled = !isNew;
        } catch (err) {
            idSelect.innerHTML = currentId ? `<option value="${currentId}" selected>${currentId}</option>` : '<option value="">加载失败</option>';
        }
    },

    renderReferences() {
        const container = document.getElementById('reference-list-container');
        const refs = App.getProp(App.state.currentEditingAvatar, 'references') || [];
        const avatarId = document.getElementById('edit-avatar-id').value;

        container.innerHTML = refs.map((r, idx) => {
            const rid = App.getProp(r, 'id') || '';
            const rname = App.getProp(r, 'name') || '';
            const rlang = App.getProp(r, 'language') || '';
            const rpath = App.getProp(r, 'audioPath') || '';
            const rtext = App.getProp(r, 'text') || '';
            const hintText = this.extractTextHint(rpath);

            return `
                <div class="reference-item" data-idx="${idx}">
                    <div class="field-row">
                        <div class="field"><label>ID</label><input type="text" value="${rid}" class="ref-id" placeholder="default"></div>
                        <div class="field"><label>名称</label><input type="text" value="${rname}" class="ref-name" placeholder="显示名称"></div>
                        <div class="field">
                            <label>语言</label>
                            <select class="ref-lang styled-select">
                                <option value="" ${rlang === '' ? 'selected' : ''}>自动/缺省</option>
                                <option value="zh" ${rlang === 'zh' ? 'selected' : ''}>中文 (zh)</option>
                                <option value="en" ${rlang === 'en' ? 'selected' : ''}>英文 (en)</option>
                                <option value="ja" ${rlang === 'ja' ? 'selected' : ''}>日文 (ja)</option>
                            </select>
                        </div>
                    </div>
                    <div class="field"><label>音频文件 (AudioPath)</label>
                        <select class="ref-path styled-select"><option value="${rpath}">${rpath || '(选择文件)'}</option></select>
                        <div class="ref-path-hint" title="点击复制文件名">${hintText}</div>
                    </div>
                    <div class="field"><label>文本 (Text)</label><input type="text" value="${rtext}" class="ref-text"></div>
                    <button type="button" class="danger-btn remove-ref-btn">移除</button>
                </div>`;
        }).join('');

        this.bindRefEvents(avatarId);
    },

    bindRefEvents(avatarId) {
        const container = document.getElementById('reference-list-container');
        if (avatarId) {
            fetch('/api/tts/fs/avatars/' + encodeURIComponent(avatarId) + '/references')
                .then(r => r.json())
                .then(files => {
                    container.querySelectorAll('.ref-path').forEach(sel => {
                        const val = sel.value;
                        sel.innerHTML = '<option value="">(选择文件)</option>' + files.map(f => `<option value="${f}"${f === val ? ' selected' : ''}>${f}</option>`).join('');
                        sel.onchange = () => {
                            const item = sel.closest('.reference-item');
                            const extracted = this.extractTextHint(sel.value);
                            item.querySelector('.ref-path-hint').textContent = extracted;
                            const textInput = item.querySelector('.ref-text');
                            if (extracted && !textInput.value.trim()) textInput.value = extracted;
                        };
                    });
                });
        }

        container.querySelectorAll('.ref-path-hint').forEach(hint => {
            hint.onclick = () => {
                const t = hint.textContent.trim();
                if (t) navigator.clipboard.writeText(t).then(() => App.showToast('已复制: ' + t));
            };
        });

        container.querySelectorAll('.ref-id').forEach(input => {
            input.oninput = () => {
                const idx = input.closest('.reference-item').dataset.idx;
                App.state.currentEditingAvatar.references[idx].id = input.value;
                this.updateDefaultRefDropdown();
            };
        });

        container.querySelectorAll('.remove-ref-btn').forEach(btn => {
            btn.onclick = () => {
                this.syncFromDOM();
                const idx = btn.closest('.reference-item').dataset.idx;
                App.state.currentEditingAvatar.references.splice(idx, 1);
                this.renderReferences();
                this.updateDefaultRefDropdown();
            };
        });
    },

    syncFromDOM() {
        const container = document.getElementById('reference-list-container');
        container.querySelectorAll('.reference-item').forEach(item => {
            const idx = item.dataset.idx;
            const ref = App.state.currentEditingAvatar.references[idx];
            if (!ref) return;
            ref.id = item.querySelector('.ref-id').value;
            ref.name = item.querySelector('.ref-name').value;
            ref.language = item.querySelector('.ref-lang').value;
            ref.audioPath = item.querySelector('.ref-path').value;
            ref.text = item.querySelector('.ref-text').value;

            // Compat for PascalCase (ensure consistency for saving)
            ref.Id = ref.id; ref.Name = ref.name; ref.Language = ref.language;
            ref.AudioPath = ref.audioPath; ref.Text = ref.text;
        });
    },

    updateDefaultRefDropdown() {
        const select = document.getElementById('avatar-form').elements.DefaultReferenceId;
        const refs = App.state.currentEditingAvatar.references || [];
        const current = App.getProp(App.state.currentEditingAvatar, 'defaultReferenceId') || 'default';
        select.innerHTML = refs.length === 0 ? '<option value="default">default</option>' :
            refs.map(r => `<option value="${r.id}"${r.id === current ? ' selected' : ''}>${r.name || r.id}</option>`).join('');
    },

    extractTextHint(filename) {
        if (!filename) return '';
        return filename.replace(/\.[^.]+$/, '').replace(/^【[^】]*】/, '').trim();
    },

    async save() {
        this.syncFromDOM();
        const form = document.getElementById('avatar-form');
        const elements = form.elements;

        const updated = {
            id: elements['Id'].value,
            name: elements['Name'].value,
            description: elements['Description'].value,
            useEngineV2: elements['UseEngineV2'].value === 'null' ? null : (elements['UseEngineV2'].value === 'true'),
            defaultReferenceId: elements['DefaultReferenceId'].value || 'default',
            references: App.state.currentEditingAvatar.references
        };

        App.showLoading('正在保存音色并重载配置...');
        try {
            const resp = await fetch('/api/tts/avatars', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updated)
            });
            if (resp.ok) {
                App.showToast('音色保存成功');
                document.getElementById('avatar-editor').style.display = 'none';
                await App.fetchAll();
            } else {
                const res = await resp.json();
                throw new Error(res.message || '保存失败');
            }
        } catch (e) { App.showToast('保存错误: ' + e.message, 'error'); }
        finally { App.hideLoading(); }
    },

    async handleImportAudio() {
        const avatarId = document.getElementById('edit-avatar-id').value;
        if (!avatarId) return App.showToast('请先选择音色 ID', 'error');

        Converter.openFilePicker('*.wav', async (filePath) => {
            try {
                App.showLoading('正在物理导入音频文件...');
                const targetDir = `resources/avatars/${avatarId}/references`;
                const resp = await fetch(`/api/tts/fs/copy-file?sourcePath=${encodeURIComponent(filePath)}&targetDir=${encodeURIComponent(targetDir)}`, { method: 'POST' });
                if (!resp.ok) throw new Error('物理复制失败');
                
                App.hideLoading();
                App.showToast('文件已成功导入到 references 目录。正在刷新列表...');
                
                // 重新绑定下拉框（刷新文件列表）
                this.bindRefEvents(avatarId);
            } catch (err) { 
                App.hideLoading(); 
                App.showToast('导入失败: ' + err.message, 'error'); 
            }
        });
    },

    async delete(id) {
        if (!confirm(`确定要删除音色 [${id}] 的配置吗？此操作不会删除物理模型文件。`)) return;
        App.showLoading('正在删除音色...');
        try {
            const resp = await fetch(`/api/tts/avatars/${id}`, { method: 'DELETE' });
            if (resp.ok) {
                App.showToast('删除成功');
                await App.fetchAll();
            } else {
                const res = await resp.json();
                throw new Error(res.message || '删除失败');
            }
        } catch (e) { App.showToast('错误: ' + e.message, 'error'); }
        finally { App.hideLoading(); }
    }
};
