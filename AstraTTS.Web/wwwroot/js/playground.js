// ============================================================
// AstraTTS WebUI — Playground Module
// 合成实验室、流式播放、并发压测
// ============================================================

const Playground = {
    audioContext: null,
    pcmPlayer: null,

    init() {
        document.getElementById('pg-play-btn').addEventListener('click', () => {
            const isStream = document.getElementById('pg-stream-toggle')?.checked;
            if (isStream) this.playStream();
            else this.playStandard();
        });

        document.getElementById('pg-speed').addEventListener('input', (e) => {
            document.getElementById('pg-speed-val').textContent = e.target.value;
        });

        // Max 2 langs
        const container = document.getElementById('pg-langs');
        if (container) {
            container.addEventListener('change', (e) => {
                if (e.target.type === 'checkbox') {
                    if (container.querySelectorAll('input:checked').length > 2) {
                        App.showToast('最多选择两种语言', 'warn');
                        e.target.checked = false;
                    }
                }
            });
        }

        // Stress Test toggle
        const stressToggle = document.getElementById('stress-test-toggle');
        if (stressToggle) {
            stressToggle.onclick = () => {
                const panel = document.getElementById('stress-test-panel');
                panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
            };
        }
        document.getElementById('run-stress-btn')?.addEventListener('click', () => this.runStressTest());
    },

    updateAvatars() {
        const select = document.getElementById('pg-avatar-select');
        const defaultId = App.getProp(App.state.config, 'defaultAvatarId');
        
        select.innerHTML = App.state.avatars.map(a => {
            const id = App.getProp(a, 'id');
            const name = App.getProp(a, 'name');
            return `<option value="${id}"${id === defaultId ? ' selected' : ''}>${name || id} (${id})</option>`;
        }).join('');

        const updateRefs = () => {
            const avatarId = select.value;
            const avatar = App.state.avatars.find(a => App.getProp(a, 'id') === avatarId);
            const refSelect = document.getElementById('pg-ref-id');
            const refs = App.getProp(avatar, 'references') || [];
            refSelect.innerHTML = '<option value="">使用默认</option>' + 
                refs.map(r => `<option value="${r.id}">${r.name || r.id}</option>`).join('');

            // Update V2 params visibility
            const globalV2 = !!App.getProp(App.state.config, 'useEngineV2');
            const isV2 = avatar ? (App.getProp(avatar, 'useEngineV2') ?? globalV2) : globalV2;
            document.getElementById('v2-params').style.display = isV2 ? 'block' : 'none';
        };

        select.onchange = updateRefs;
        updateRefs();
    },

    getParams() {
        const avatarId = document.getElementById('pg-avatar-select').value;
        const avatar = App.state.avatars.find(a => App.getProp(a, 'id') === avatarId);
        const globalV2 = !!App.getProp(App.state.config, 'useEngineV2');
        const isV2 = avatar ? (App.getProp(avatar, 'useEngineV2') ?? globalV2) : globalV2;
        const selectedLangs = Array.from(document.querySelectorAll('#pg-langs input:checked')).map(i => i.value);

        const params = {
            text: document.getElementById('pg-text').value,
            avatarId: avatarId,
            referenceId: document.getElementById('pg-ref-id').value || null,
            speed: parseFloat(document.getElementById('pg-speed').value),
            languages: selectedLangs.length > 0 ? selectedLangs : null
        };

        if (isV2) {
            params.temperature = parseFloat(document.getElementById('pg-temp').value);
            params.noiseScale = parseFloat(document.getElementById('pg-noise').value);
            params.topK = parseInt(document.getElementById('pg-topk').value);
        }
        return params;
    },

    async playStandard() {
        const params = this.getParams();
        if (!params.text) return App.showToast('请输入合成文本', 'error');

        document.getElementById('pg-play-btn').disabled = true;
        document.getElementById('playing-status').style.display = 'block';
        document.getElementById('synth-perf-panel').style.display = 'none';
        document.getElementById('pg-download-area').innerHTML = '';

        const startTime = performance.now();
        try {
            const resp = await fetch('/api/tts/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            if (!resp.ok) throw new Error('合成失败');

            this.updateMetrics(resp, performance.now() - startTime);

            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            new Audio(url).play();

            // 下载链接
            const ts = new Date().toTimeString().slice(0, 8).replace(/:/g, '');
            document.getElementById('pg-download-area').innerHTML =
                `<a href="${url}" download="tts_${ts}.wav" class="download-link">💾 下载本次合成音频 (WAV)</a>`;

            App.showToast('合成成功，开始播放');
        } catch (e) { App.showToast('合成错误: ' + e.message, 'error'); }
        finally {
            document.getElementById('pg-play-btn').disabled = false;
            document.getElementById('playing-status').style.display = 'none';
        }
    },

    async playStream() {
        const params = this.getParams();
        if (!params.text) return App.showToast('请输入合成文本', 'error');

        document.getElementById('pg-play-btn').disabled = true;
        document.getElementById('playing-status').style.display = 'block';
        document.getElementById('playing-status').textContent = '正在初始化流式音频...';
        document.getElementById('pg-download-area').innerHTML = '';
        document.getElementById('synth-perf-panel').style.display = 'none';

        try {
            const query = new URLSearchParams();
            for (let k in params) {
                if (params[k] !== null) {
                   if (Array.isArray(params[k])) params[k].forEach(v => query.append(k, v));
                   else query.append(k, params[k]);
                }
            }

            const startTime = performance.now();
            const response = await fetch('/api/tts/predict-stream?' + query.toString());
            if (!response.ok) throw new Error('流式请求失败');

            // 从响应头读取真实采样率，避免硬编码导致音调变形
            const sampleRate = parseInt(response.headers.get('X-Audio-Sample-Rate')) || 32000;

            // 如果采样率变化或首次调用，(重新)创建 AudioContext
            if (!this.audioContext || this.audioContext.sampleRate !== sampleRate) {
                if (this.audioContext) this.audioContext.close();
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
            }
            if (this.audioContext.state === 'suspended') await this.audioContext.resume();

            const reader = response.body.getReader();
            let firstChunkTime = 0;
            let firstChunk = true;
            let totalChunks = 0;
            let totalSamples = 0;
            let nextStartTime = this.audioContext.currentTime;
            const allChunks = [];

            // PCM float32 字节对齐缓冲：ReadableStream 的分块可能在
            // 4 字节边界之外截断，导致 Float32Array 构造失败或数据错位
            let residual = new Uint8Array(0);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                // 将残留字节与新数据拼接
                let combined;
                if (residual.length > 0) {
                    combined = new Uint8Array(residual.length + value.length);
                    combined.set(residual);
                    combined.set(value, residual.length);
                } else {
                    combined = value;
                }

                // 取 4 字节对齐的部分，剩余存入 residual
                const alignedLen = combined.length - (combined.length % 4);
                residual = combined.slice(alignedLen);

                if (alignedLen === 0) continue;

                if (firstChunk) {
                    firstChunkTime = performance.now() - startTime;
                    document.getElementById('perf-fbl').textContent = Math.round(firstChunkTime);
                    document.getElementById('synth-perf-panel').style.display = 'flex';
                    firstChunk = false;
                }

                // 安全构造 Float32Array：使用 DataView 逐样本提取以确保字节序正确
                const sampleCount = alignedLen / 4;
                const floatData = new Float32Array(sampleCount);
                const dataView = new DataView(combined.buffer, combined.byteOffset, alignedLen);
                for (let i = 0; i < sampleCount; i++) {
                    floatData[i] = dataView.getFloat32(i * 4, true); // little-endian
                }

                allChunks.push(floatData);
                totalSamples += sampleCount;

                const audioBuffer = this.audioContext.createBuffer(1, floatData.length, sampleRate);
                audioBuffer.getChannelData(0).set(floatData);

                const source = this.audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(this.audioContext.destination);
                const startAt = Math.max(nextStartTime, this.audioContext.currentTime);
                source.start(startAt);
                nextStartTime = startAt + audioBuffer.duration;

                totalChunks++;
                document.getElementById('playing-status').textContent = `正在播放流 (${sampleRate}Hz): 已接收 ${totalChunks} 块...`;
            }

            // 流式性能指标
            const totalElapsed = performance.now() - startTime;
            const audioDuration = totalSamples / sampleRate;
            const rtf = (totalElapsed / 1000) / audioDuration;
            const speed = (audioDuration / (totalElapsed / 1000)).toFixed(2);
            document.getElementById('perf-fbl-label').textContent = '首包延迟';
            document.getElementById('perf-fbl-unit').textContent = 'ms';
            document.getElementById('perf-speed').textContent = speed;
            document.getElementById('perf-speed-unit').textContent = 'x';
            document.getElementById('perf-rtf').textContent = rtf.toFixed(3);

            // 合成 WAV 供下载
            const wavBlob = this.pcmToWav(allChunks, totalSamples, sampleRate);
            const wavUrl = URL.createObjectURL(wavBlob);
            const ts = new Date().toTimeString().slice(0, 8).replace(/:/g, '');
            document.getElementById('pg-download-area').innerHTML =
                `<a href="${wavUrl}" download="tts_stream_${ts}.wav" class="download-link">💾 下载流式合成音频 (WAV, ${audioDuration.toFixed(1)}s)</a>`;

            App.showToast('流式合成完毕');
        } catch (e) { App.showToast('流式错误: ' + e.message, 'error'); }
        finally {
            document.getElementById('pg-play-btn').disabled = false;
            document.getElementById('playing-status').style.display = 'none';
            document.getElementById('playing-status').textContent = '正在处理音频流...';
        }
    },

    updateMetrics(resp, fbl) {
        const rtf = resp.headers.get('X-RTF');
        const totalTime = resp.headers.get('X-Synthesis-Time');
        const tokenCount = resp.headers.get('X-Token-Count');

        // 标准模式标签
        document.getElementById('perf-fbl-label').textContent = '生成耗时';
        document.getElementById('perf-fbl-unit').textContent = 'ms';
        document.getElementById('perf-speed-unit').textContent = 'tokens/s';
        document.getElementById('perf-fbl').textContent = Math.round(fbl);
        if (rtf) document.getElementById('perf-rtf').textContent = rtf;
        
        if (tokenCount && totalTime) {
            const speed = (parseInt(tokenCount) / (parseFloat(totalTime) / 1000.0)).toFixed(1);
            document.getElementById('perf-speed').textContent = speed;
        } else if (rtf) {
            document.getElementById('perf-speed').textContent = (1.0 / parseFloat(rtf)).toFixed(2);
        }
        document.getElementById('synth-perf-panel').style.display = 'flex';
    },

    /** 将 Float32 PCM 块数组合成为 WAV Blob (16-bit PCM) */
    pcmToWav(chunks, totalSamples, sampleRate) {
        const buffer = new ArrayBuffer(44 + totalSamples * 2);
        const view = new DataView(buffer);
        const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
        const dataSize = totalSamples * 2;
        writeStr(0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeStr(8, 'WAVE');
        writeStr(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeStr(36, 'data');
        view.setUint32(40, dataSize, true);
        let offset = 44;
        for (const chunk of chunks) {
            for (let i = 0; i < chunk.length; i++) {
                const s = Math.max(-1, Math.min(1, chunk[i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                offset += 2;
            }
        }
        return new Blob([buffer], { type: 'audio/wav' });
    },

    // --- Stress Test ---

    async runStressTest() {
        const concurrency = parseInt(document.getElementById('stress-concurrency').value) || 1;
        const repeats = parseInt(document.getElementById('stress-repeats').value) || 1;
        
        // 使用压测专用文本（如果有），否则回退到 Playground 文本
        const stressText = document.getElementById('stress-text')?.value?.trim();
        const baseText = stressText || document.getElementById('pg-text').value || "测试并发。";
        
        // 解析多行文本（每行一个测试用例，轮询使用）
        const textLines = baseText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
        
        // 支持多音色轮询
        const stressAvatarMode = document.getElementById('stress-avatar-mode')?.value || 'current';
        let avatarIds = [];
        if (stressAvatarMode === 'all') {
            avatarIds = App.state.avatars.map(a => App.getProp(a, 'id'));
        } else {
            avatarIds = [document.getElementById('pg-avatar-select').value];
        }

        // 参考音频
        const stressRefId = document.getElementById('stress-ref-id')?.value || '';
        
        const resultsTable = document.getElementById('stress-results-body');
        resultsTable.innerHTML = '';
        document.getElementById('stress-stats').innerHTML = '正在初始化...';
        document.getElementById('run-stress-btn').disabled = true;
        
        const tasks = [];
        let completed = 0;
        let successful = 0;
        let totalLatency = 0;
        let minLatency = Infinity;
        let maxLatency = 0;
        const startTime = performance.now();

        for (let i = 0; i < repeats; i++) {
            const text = textLines[i % textLines.length];
            const avatarId = avatarIds[i % avatarIds.length];

            tasks.push(async (id) => {
                const row = document.createElement('tr');
                row.innerHTML = `<td>#${id}</td><td>${avatarId}</td><td class="status-pending">执行中...</td><td>-</td><td>-</td><td>-</td>`;
                resultsTable.prepend(row);
                
                const body = { text, avatarId };
                if (stressRefId) body.referenceId = stressRefId;
                
                const reqStart = performance.now();
                try {
                    const resp = await fetch('/api/tts/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                    const duration = performance.now() - reqStart;
                    if (resp.ok) {
                        successful++;
                        totalLatency += duration;
                        if (duration < minLatency) minLatency = duration;
                        if (duration > maxLatency) maxLatency = duration;
                        row.cells[2].textContent = '成功';
                        row.cells[2].className = 'status-success';
                        row.cells[3].textContent = Math.round(duration) + 'ms';
                        row.cells[4].textContent = (resp.headers.get('X-Audio-Duration') || '-') + 's';
                        // 下载链接
                        const blob = await resp.blob();
                        const url = URL.createObjectURL(blob);
                        row.cells[5].innerHTML = `<a href="${url}" download="stress_${id}.wav" title="下载">💾</a>`;
                    } else {
                        row.cells[2].textContent = `失败 (${resp.status})`;
                        row.cells[2].className = 'status-fail';
                        row.cells[3].textContent = Math.round(duration) + 'ms';
                    }
                } catch (e) {
                    row.cells[2].textContent = '异常';
                    row.cells[2].className = 'status-fail';
                    row.cells[3].textContent = e.message;
                }
                completed++;
                this.updateStressStats(startTime, completed, repeats, successful, totalLatency, minLatency, maxLatency);
            });
        }

        const limit = async (concurrency, tasks) => {
            const pool = new Set();
            for (let i = 0; i < tasks.length; i++) {
                const promise = tasks[i](i + 1);
                pool.add(promise);
                promise.then(() => pool.delete(promise));
                if (pool.size >= concurrency) await Promise.race(pool);
            }
            await Promise.all(pool);
        };

        App.showToast(`压测开始：并发 ${concurrency}, 总数 ${repeats}, 文本 ${textLines.length} 条, 音色 ${avatarIds.length} 个`);
        await limit(concurrency, tasks);
        document.getElementById('run-stress-btn').disabled = false;
        App.showToast('压测完成');
    },

    updateStressStats(startTime, completed, total, successful, totalLatency, minLat, maxLat) {
        const elapsed = (performance.now() - startTime) / 1000;
        const avgLat = successful > 0 ? (totalLatency / successful).toFixed(0) : '-';
        const qps = (completed / elapsed).toFixed(2);
        const failed = completed - successful;
        const minStr = minLat < Infinity ? Math.round(minLat) : '-';
        const maxStr = maxLat > 0 ? Math.round(maxLat) : '-';
        document.getElementById('stress-stats').innerHTML = 
            `<span>进度: <strong>${completed}/${total}</strong></span> | ` +
            `<span style="color:var(--online-color)">成功: ${successful}</span>` +
            (failed > 0 ? ` | <span style="color:#ef4444">失败: ${failed}</span>` : '') +
            ` | 平均: ${avgLat}ms | 最小: ${minStr}ms | 最大: ${maxStr}ms` +
            ` | QPS: <strong>${qps}</strong> | 总耗时: ${elapsed.toFixed(1)}s`;
    }
};
