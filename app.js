import { Wllama } from 'https://unpkg.com/@wllama/wllama/esm/index.js';

// ─── WASM config ──────────────────────────────────────────────────
const CONFIG_PATHS = {
  'default': 'https://unpkg.com/@wllama/wllama/esm/wasm/wllama.wasm',
};

let HF_REPO = 'asubah/Qwen3-GGUF';

// ─── DOM refs ─────────────────────────────────────────────────────
const $repoInput   = document.getElementById('model-repo-input');
const $detectBtn   = document.getElementById('detect-btn');
const $sel         = document.getElementById('model-select');
const $selWrap     = document.getElementById('model-select-wrap');
const $loadBtn     = document.getElementById('load-btn');
const $gpuBtn      = document.getElementById('gpu-btn');
const $stopBtn     = document.getElementById('stop-btn');
const $clearChat   = document.getElementById('clear-chat-btn');
const $clearCache  = document.getElementById('clear-cache-btn');
const $chat        = document.getElementById('chat');
const $empty       = document.getElementById('empty');
const $input       = document.getElementById('user-input');
const $sendBtn     = document.getElementById('send-btn');
const $status      = document.getElementById('status');
const $progWrap    = document.getElementById('prog-wrap');
const $progBar     = document.getElementById('prog-bar');

// ─── App state ────────────────────────────────────────────────────
let wllama       = null;
let isLoaded     = false;
let isGenerating = false;
let history      = [];
let abortCtrl    = null;

// ─── WebGPU state ─────────────────────────────────────────────────
const GPU_SUPPORTED = (() => {
  try { return new Wllama(CONFIG_PATHS).isSupportWebGPU(); }
  catch (_) { return false; }
})();

const IS_FIREFOX = navigator.userAgent.toLowerCase().includes('firefox');

let gpuEnabled = GPU_SUPPORTED;

if (GPU_SUPPORTED) {
  $gpuBtn.style.display = 'inline-flex';
  syncGpuBtn();
}

function syncGpuBtn() {
  if (!GPU_SUPPORTED) return;
  if (gpuEnabled) {
    $gpuBtn.textContent = IS_FIREFOX ? '🟡 WebGPU: ON (compat)' : '🚀 WebGPU: ON';
    $gpuBtn.classList.remove('border-bd', 'text-muted');
    $gpuBtn.classList.add('border-violet-700', 'text-violet-400');
  } else {
    $gpuBtn.textContent = '🧵 WebGPU: OFF';
    $gpuBtn.classList.remove('border-violet-700', 'text-violet-400');
    $gpuBtn.classList.add('border-bd', 'text-muted');
  }
}

// ─── Status helpers ───────────────────────────────────────────────
function setStatus(html, cls = '') {
  $status.innerHTML = html;
  $status.classList.remove('text-muted', 'text-green-400', 'text-red-400', 'text-amber-400');
  const map = { ok: 'text-green-400', err: 'text-red-400', busy: 'text-amber-400' };
  if (map[cls]) $status.classList.add(map[cls]);
}

function setProgress(pct) {
  $progWrap.classList.toggle('hidden', !(pct > 0 && pct < 100));
  $progBar.style.width = pct + '%';
}

// ─── HTML helpers ─────────────────────────────────────────────────
function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function toHtml(raw) {
  const parts = [];
  let pos = 0;
  const re = /<think>([\s\S]*?)<\/think>/g;
  let m;
  while ((m = re.exec(raw)) !== null) {
    if (m.index > pos)
      parts.push(`<span>${esc(raw.slice(pos, m.index))}</span>`);
    parts.push(
      `<span class="block bg-surface2 border-l-2 border-bd rounded-r-md px-3 py-1.5 mb-2 ` +
      `text-muted text-[12.5px] italic whitespace-pre-wrap">💭 ${esc(m[1].trim())}</span>`
    );
    pos = m.index + m[0].length;
  }
  if (pos < raw.length) parts.push(`<span>${esc(raw.slice(pos))}</span>`);
  return parts.join('');
}

function appendBubble(role, htmlContent) {
  $empty.style.display = 'none';
  const isUser = role === 'user';
  const row    = document.createElement('div');
  row.className = `flex gap-2.5 w-full max-w-2xl ${isUser ? 'ml-auto flex-row-reverse' : 'mr-auto'}`;
  row.innerHTML = `
    <div class="w-7 h-7 rounded-full text-[11px] font-bold flex items-center justify-center shrink-0 select-none
                ${isUser ? 'bg-violet-600 text-white' : 'bg-surface2 text-muted border border-bd'}">
      ${isUser ? 'You' : 'AI'}
    </div>
    <div class="msg-body border rounded-xl px-3.5 py-2.5 text-[13.5px] leading-relaxed
                break-words max-w-[calc(100%-2.5rem)] whitespace-pre-wrap
                ${isUser ? 'bg-[#0d2d5e] border-blue-800' : 'bg-surface border-bd'}">
      ${htmlContent}
    </div>`;
  $chat.appendChild(row);
  $chat.scrollTop = $chat.scrollHeight;
  return row.querySelector('.msg-body');
}

function wipeMessages() {
  $chat.querySelectorAll('.flex.gap-2\\.5').forEach(el => el.remove());
  $empty.style.display = '';
}

function setGenerating(on) {
  isGenerating           = on;
  $stopBtn.style.display = on ? 'inline-flex' : 'none';
  $sendBtn.disabled      = on;
  $input.disabled        = on;
  $gpuBtn.disabled       = on;
}

// ─── Detect GGUF files from HF repo ───────────────────────────────
async function detectGGUFFiles() {
  const repo = $repoInput.value.trim();
  if (!repo) {
    setStatus('⚠️ Enter a HuggingFace model ID (e.g., meta-llama/Llama-2-7b-hf)', 'busy');
    return;
  }

  $detectBtn.disabled = true;
  setStatus(`⏳ Scanning ${repo} for GGUF files…`, 'busy');

  try {
    const apiUrl = `https://huggingface.co/api/models/${repo}`;
    const response = await fetch(apiUrl);
    if (!response.ok) throw new Error('Model not found or not accessible');

    const modelData = await response.json();
    const siblings = modelData.siblings || [];

    // Filter GGUF files
    const ggufFiles = siblings.filter(f => f.rfilename.endsWith('.gguf'));

    if (ggufFiles.length === 0) {
      setStatus(
        `✗ No GGUF files found in ${repo}. ` +
        `<span class="text-[11px] block mt-1 text-muted">💡 Convert this model using ` +
        `<a href="https://github.com/ggml-org/llama.cpp" target="_blank" class="text-violet-400 hover:underline">llama.cpp</a> ` +
        `or use a repo with pre-converted GGUF files.</span>`,
        'err'
      );
      $selWrap.style.display = 'none';
      $loadBtn.disabled = true;
      return;
    }

    HF_REPO = repo;

    // Group by base name (handle shards)
    const models = new Map();
    ggufFiles.forEach(f => {
      const name = f.rfilename;
      const match = name.match(/^(.+?)-\d+-of-\d+\.gguf$/);
      const baseName = match ? match[1] : name.replace('.gguf', '');
      if (!models.has(baseName)) models.set(baseName, []);
      models.get(baseName).push(name);
    });

    // Populate select
    $sel.innerHTML = '';
    const sorted = Array.from(models.entries()).sort((a, b) => a[0].localeCompare(b[0]));
    sorted.forEach(([baseName, files]) => {
      const totalSize = files.length * 500; // rough estimate in MB
      const shardInfo = files.length > 1 ? ` (${files.length} shards)` : '';
      const option = document.createElement('option');
      option.value = files[0]; // first shard filename
      option.textContent = `${baseName}${shardInfo} ~${Math.round(totalSize / 1024)}GB`;
      $sel.appendChild(option);
    });

    $selWrap.style.display = 'block';
    $loadBtn.disabled = false;
    setStatus(`✓ Found ${ggufFiles.length} GGUF file(s) · Select a model and click Load`, 'ok');

  } catch (err) {
    setStatus(`✗ Detection failed: ${esc(err.message)}`, 'err');
    $selWrap.style.display = 'none';
    $loadBtn.disabled = true;
  } finally {
    $detectBtn.disabled = false;
  }
}

// ─── Load model ───────────────────────────────────────────────────
async function loadModel() {
  const file = $sel.value;
  if (!file) {
    setStatus('Select a model first', 'err');
    return;
  }

  $loadBtn.disabled = true;
  $gpuBtn.disabled  = true;
  $detectBtn.disabled = true;
  $sel.disabled     = true;

  if (wllama) {
    try { await wllama.exit(); } catch (_) {}
    wllama   = null;
    isLoaded = false;
  }

  history = [];
  wipeMessages();

  wllama = new Wllama(CONFIG_PATHS, { parallelDownloads: 5 });

  if (GPU_SUPPORTED && gpuEnabled && IS_FIREFOX) {
    wllama.setCompat('default', 'firefox_safari');
  } else if (GPU_SUPPORTED && gpuEnabled) {
    wllama.setCompat('default');
  }

  const backendLabel = !GPU_SUPPORTED
    ? '🧵 CPU'
    : gpuEnabled
      ? (IS_FIREFOX ? '🟡 WebGPU (compat)' : '🚀 WebGPU')
      : '🧵 CPU (GPU off)';

  setStatus(`⌛ Downloading <b>${file}</b>… <span class="opacity-60">${backendLabel}</span>`, 'busy');
  setProgress(1);

  try {
    await wllama.loadModelFromHF(
      { repo: HF_REPO, file },
      {
        n_ctx: 4096,
        n_gpu_layers: GPU_SUPPORTED && gpuEnabled ? 99999 : 0,
        progressCallback({ loaded, total }) {
          const pct = total > 0 ? Math.round(loaded / total * 100) : 0;
          setProgress(pct);
          setStatus(
            `⌛ Downloading <b>${file}</b>… ${pct}% <span class="opacity-60">${backendLabel}</span>`,
            'busy'
          );
        },
      }
    );

    setProgress(0);
    isLoaded = true;
    $input.disabled      = false;
    $sendBtn.disabled    = false;
    $clearChat.disabled  = false;
    $loadBtn.textContent = '↺ Reload';
    setStatus(`✓ <b>${file}</b> ready · ${backendLabel}`, 'ok');
    $input.focus();

  } catch (err) {
    setProgress(0);
    setStatus(`✗ Load failed: ${esc(err.message)}`, 'err');
  } finally {
    $loadBtn.disabled = false;
    $gpuBtn.disabled  = false;
    $detectBtn.disabled = false;
    $sel.disabled     = false;
  }
}

// ─── GPU toggle ───────────────────────────────────────────────────
function toggleGpu() {
  if (isGenerating) return;
  gpuEnabled = !gpuEnabled;
  syncGpuBtn();

  if (isLoaded) {
    setStatus(
      `⚠️ Backend changed to <b>${gpuEnabled ? 'WebGPU' : 'CPU'}</b> — ` +
      `click <b>↺ Reload</b> to apply.`,
      'busy'
    );
  }
}

// ─── Send & stream ────────────────────────────────────────────────
async function sendMessage() {
  const text = $input.value.trim();
  if (!text || !isLoaded || isGenerating) return;

  setGenerating(true);
  $input.value = '';
  resizeInput();

  history.push({ role: 'user', content: text + ' /no_think' });

  appendBubble('user', toHtml(text));
  const $body = appendBubble('assistant', '<span class="cursor"></span>');
  let reply = '';

  abortCtrl = new AbortController();

  try {
    const iter = await wllama.createChatCompletion({
      messages:    [{ role: 'system', content: 'You are a helpful assistant.' }, ...history],
      max_tokens:  1024,
      temperature: 0.7,
      top_p:       0.9,
      stream:      true,
      abortSignal: abortCtrl.signal,
    });

    for await (const chunk of iter) {
      reply += chunk.choices[0]?.delta?.content ?? '';
      $body.innerHTML = toHtml(reply) + '<span class="cursor"></span>';
      $chat.scrollTop = $chat.scrollHeight;
    }

    $body.innerHTML = toHtml(reply);
    history.push({ role: 'assistant', content: reply });

  } catch (err) {
    const aborted = err.name === 'AbortError' || abortCtrl.signal.aborted;
    if (aborted) {
      $body.innerHTML =
        (reply ? toHtml(reply) : '<em class="opacity-50">…</em>') +
        '<br><em class="text-muted text-xs">(stopped)</em>';
      if (reply) history.push({ role: 'assistant', content: reply });
    } else {
      $body.innerHTML = `<em class="text-red-400">Error: ${esc(err.message)}</em>`;
    }
  } finally {
    setGenerating(false);
    $input.focus();
  }
}

// ─── Clear cache ──────────────────────────────────────────────────
async function clearCache() {
  if (!confirm('Delete all cached model files from OPFS?\nThey will re-download on next use.')) return;

  $clearCache.disabled = true;

  if (wllama && isLoaded) {
    try { await wllama.exit(); } catch (_) {}
    wllama   = null;
    isLoaded = false;
    $input.disabled      = true;
    $sendBtn.disabled    = true;
    $clearChat.disabled  = true;
    $loadBtn.textContent = 'Load Model';
  }

  try {
    const root = await navigator.storage.getDirectory();
    for await (const name of root.keys()) {
      try { await root.removeEntry(name, { recursive: true }); } catch (_) { }
    }

    if ('caches' in window) {
      for (const key of await caches.keys()) await caches.delete(key);
    }

    if ('indexedDB' in window) {
      const dbs = await indexedDB.databases();
      await Promise.all(
        dbs
          .filter(db => db.name?.match(/wllama|localforage/i))
          .map(db => new Promise(res => {
            const req = indexedDB.deleteDatabase(db.name);
            req.onsuccess = req.onerror = res;
          }))
      );
    }

    setStatus('🗑 Cache cleared — models will re-download on next load.', 'ok');
    setTimeout(() => location.reload(), 1200);
  } catch (err) {
    setStatus(`✗ Cache clear failed: ${esc(err.message)}`, 'err');
  } finally {
    $clearCache.disabled = false;
  }
}

// ─── Textarea auto-resize ─────────────────────────────────────────
function resizeInput() {
  $input.style.height = 'auto';
  $input.style.height = Math.min($input.scrollHeight, 144) + 'px';
}

// ─── Event wiring ─────────────────────────────────────────────────
$detectBtn .addEventListener('click',   detectGGUFFiles);
$repoInput .addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); detectGGUFFiles(); }
});
$loadBtn   .addEventListener('click',   loadModel);
$gpuBtn    .addEventListener('click',   toggleGpu);
$stopBtn   .addEventListener('click',   () => abortCtrl?.abort());
$clearChat .addEventListener('click',   () => { history = []; wipeMessages(); setStatus('Chat history cleared.'); });
$clearCache.addEventListener('click',   clearCache);
$sendBtn   .addEventListener('click',   sendMessage);
$input     .addEventListener('input',   resizeInput);
$input     .addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ─── Initialize with default repo ─────────────────────────────────
$repoInput.value = HF_REPO;
detectGGUFFiles();