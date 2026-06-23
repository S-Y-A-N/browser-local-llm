import { Wllama } from 'https://unpkg.com/@wllama/wllama/esm/index.js';

// ─── WASM config ──────────────────────────────────────────────────
const CONFIG_PATHS = {
  'default': 'https://unpkg.com/@wllama/wllama/esm/wasm/wllama.wasm',
};

let HF_REPO = 'asubah/Qwen3-GGUF';

// ─── DOM refs ─────────────────────────────────────────────────────
const $repoInput      = document.getElementById('model-repo-input');
const $detectBtn      = document.getElementById('detect-btn');
const $sel            = document.getElementById('model-select');
const $selWrap        = document.getElementById('model-select-wrap');
const $loadBtn        = document.getElementById('load-btn');
const $gpuBtn         = document.getElementById('gpu-btn');
const $gpuBtnM        = document.getElementById('gpu-btn-m');
const $stopBtn        = document.getElementById('stop-btn');
const $stopBtnM       = document.getElementById('stop-btn-m');
const $clearChat      = document.getElementById('clear-chat-btn');
const $clearChatM     = document.getElementById('clear-chat-btn-m');
const $clearCache     = document.getElementById('clear-cache-btn');
const $clearCacheM    = document.getElementById('clear-cache-btn-m');
const $chat           = document.getElementById('chat');
const $empty          = document.getElementById('empty');
const $input          = document.getElementById('user-input');
const $sendBtn        = document.getElementById('send-btn');
const $status         = document.getElementById('status');
const $progWrap       = document.getElementById('prog-wrap');
const $progBar        = document.getElementById('prog-bar');
const $sidebar        = document.getElementById('sidebar');
const $sidebarOverlay = document.getElementById('sidebar-overlay');
const $chatList       = document.getElementById('chat-list');
const $sidebarToggle  = document.getElementById('sidebar-toggle');
const $newChatBtn     = document.getElementById('new-chat-btn');
const $themeBtn       = document.getElementById('theme-btn');
const $themeIconDark  = document.getElementById('theme-icon-dark');
const $themeIconLight = document.getElementById('theme-icon-light');

// ─── App state ────────────────────────────────────────────────────
let wllama       = null;
let isLoaded     = false;
let isGenerating = false;
let history      = [];
let abortCtrl    = null;
let cachedFiles  = new Set();

// ─── Theme ────────────────────────────────────────────────────────
let theme = localStorage.getItem('wllama_theme') || 'dark';

function applyTheme(t) {
  theme = t;
  document.documentElement.setAttribute('data-theme', t);
  localStorage.setItem('wllama_theme', t);
  $themeIconDark.style.display  = t === 'dark'  ? '' : 'none';
  $themeIconLight.style.display = t === 'light' ? '' : 'none';
}

applyTheme(theme);
$themeBtn.addEventListener('click', () => applyTheme(theme === 'dark' ? 'light' : 'dark'));

// ─── WebGPU state ─────────────────────────────────────────────────
const GPU_SUPPORTED = (() => {
  try { return new Wllama(CONFIG_PATHS).isSupportWebGPU(); }
  catch (_) { return false; }
})();

// Deep WebGPU diagnostics at startup
(async () => {
  console.group('[wllama] WebGPU startup diagnostics');
  console.log('navigator.gpu present:', !!navigator.gpu);
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        const info = adapter.info ?? {};
        console.log('Adapter vendor:',       info.vendor      ?? '(unknown)');
        console.log('Adapter device:',       info.device      ?? '(unknown)');
        console.log('Adapter description:',  info.description ?? '(unknown)');
        console.log('Adapter architecture:', info.architecture ?? '(unknown)');
        console.log('Adapter backend:',      info.backend     ?? '(unknown)');
        const device = await adapter.requestDevice();
        console.log('Device obtained:', !!device);
        console.log('Device limits (maxBufferSize):', device.limits.maxBufferSize);
        device.destroy();
      } else {
        console.warn('requestAdapter() returned null — GPU may be blocked or unavailable');
      }
    } catch (e) {
      console.error('WebGPU adapter/device request failed:', e);
    }
  } else {
    console.warn('navigator.gpu is undefined — WebGPU not exposed in this context');
  }
  console.log('wllama isSupportWebGPU():', GPU_SUPPORTED);
  console.groupEnd();
})();

const IS_FIREFOX = navigator.userAgent.toLowerCase().includes('firefox');
let gpuEnabled = GPU_SUPPORTED;

function setGpuBtnVisible(visible) {
  $gpuBtn.style.display  = visible ? 'inline-flex' : 'none';
  $gpuBtnM.style.display = visible ? 'inline-flex' : 'none';
}

if (GPU_SUPPORTED) {
  setGpuBtnVisible(true);
  syncGpuBtn();
}

function syncGpuBtn() {
  if (!GPU_SUPPORTED) return;
  const label = gpuEnabled
    ? (IS_FIREFOX ? '🟡 WebGPU: ON' : '🚀 WebGPU: ON')
    : '🧵 WebGPU: OFF';
  [$gpuBtn, $gpuBtnM].forEach($b => {
    $b.textContent = label;
    if (gpuEnabled) {
      $b.style.color       = '#a78bfa';
      $b.style.borderColor = '#7c3aed';
    } else {
      $b.style.color       = '';
      $b.style.borderColor = '';
    }
  });
}

// ─── Status helpers ───────────────────────────────────────────────
const STATUS_COLORS = { ok: '#22c55e', err: '#ef4444', busy: '#f59e0b' };

function setStatus(html, cls = '') {
  $status.innerHTML = html;
  $status.style.color = cls ? (STATUS_COLORS[cls] || '') : 'var(--muted)';
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
    parts.push(`<span class="think-block">💭 ${esc(m[1].trim())}</span>`);
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
                ${isUser ? 'avatar-user' : 'avatar-ai'}">
      ${isUser ? 'You' : 'AI'}
    </div>
    <div class="msg-body border rounded-xl px-3.5 py-1 text-[13.5px] leading-relaxed
                break-words max-w-[calc(100%-2.5rem)] whitespace-pre-wrap w-fit
                ${isUser ? 'bubble-user' : 'bubble-ai'}">
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
  isGenerating = on;
  [$stopBtn, $stopBtnM].forEach($b => { $b.style.display = on ? 'inline-flex' : 'none'; });
  $sendBtn.disabled = on;
  $input.disabled   = on;
  [$gpuBtn, $gpuBtnM].forEach($b => { $b.disabled = on; });
}

// ─── Sidebar ──────────────────────────────────────────────────────
let sidebarOpen = window.innerWidth >= 640; // open by default on desktop

function setSidebar(open) {
  sidebarOpen = open;
  if (window.innerWidth < 640) {
    $sidebar.classList.toggle('open', open);
  } else {
    $sidebar.classList.toggle('collapsed', !open);
  }
}

setSidebar(sidebarOpen);

$sidebarToggle.addEventListener('click', () => setSidebar(!sidebarOpen));

// Close sidebar overlay on mobile when clicking it
$sidebarOverlay.addEventListener('click', () => setSidebar(false));

// ─── Saved chats (localStorage) ───────────────────────────────────
const STORAGE_KEY = 'wllama_chats';
let activeChatId  = null;

function loadAllChats() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
  catch (_) { return []; }
}

function saveAllChats(chats) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
}

function upsertChat(id, patch) {
  const chats = loadAllChats();
  const idx   = chats.findIndex(c => c.id === id);
  if (idx >= 0) {
    chats[idx] = { ...chats[idx], ...patch, updatedAt: Date.now() };
  } else {
    chats.unshift({ id, ...patch, updatedAt: Date.now() });
  }
  saveAllChats(chats);
  renderChatList();
}

function deleteChat(id) {
  const chats = loadAllChats().filter(c => c.id !== id);
  saveAllChats(chats);
  if (activeChatId === id) {
    activeChatId = null;
    history = [];
    wipeMessages();
    setStatus('Chat deleted.');
  }
  renderChatList();
}

function renderChatList() {
  const chats = loadAllChats();
  $chatList.innerHTML = '';

  if (chats.length === 0) {
    $chatList.innerHTML =
      `<li class="px-3 py-3 text-[12px] italic" style="color:var(--muted)">No saved chats yet.</li>`;
    return;
  }

  chats.forEach(chat => {
    const isActive = chat.id === activeChatId;
    const li = document.createElement('li');
    li.className = `chat-item group flex items-start gap-1.5 px-2.5 py-2 cursor-pointer
                    rounded-md mx-1 my-0.5 transition-colors ${isActive ? 'active' : ''}`;

    li.innerHTML = `
      <div class="flex-1 min-w-0">
        <div class="text-[12.5px] font-medium truncate leading-snug" style="color:var(--text)">${esc(chat.title)}</div>
        <div class="text-[11px] truncate mt-0.5" style="color:var(--muted)" title="${esc(chat.model.file)}">${esc(chat.model.file)}</div>
      </div>
      <button class="del-btn shrink-0 opacity-0 group-hover:opacity-100 transition-opacity mt-0.5
                     leading-none text-[16px] hover:text-red-400" style="color:var(--muted)" title="Delete">×</button>`;

    li.addEventListener('click', () => {
      openSavedChat(chat.id);
      if (window.innerWidth < 640) setSidebar(false);
    });
    li.querySelector('.del-btn').addEventListener('click', e => {
      e.stopPropagation();
      if (confirm('Delete this chat?')) deleteChat(chat.id);
    });

    $chatList.appendChild(li);
  });
}

function persistCurrentChat() {
  if (history.length === 0 || !isLoaded) return;
  const file = $sel.value;
  const repo = HF_REPO;
  if (!file) return;

  const firstUser = history.find(m => m.role === 'user');
  const rawTitle  = firstUser ? firstUser.content.replace(/ \/no_think$/, '') : 'Chat';
  const title     = rawTitle.length > 48 ? rawTitle.slice(0, 48) + '…' : rawTitle;

  if (!activeChatId) activeChatId = `chat_${Date.now()}`;
  upsertChat(activeChatId, { title, model: { repo, file }, messages: history });
}

async function openSavedChat(id) {
  if (isGenerating) return;
  const chats = loadAllChats();
  const chat  = chats.find(c => c.id === id);
  if (!chat) return;

  activeChatId = id;
  renderChatList();

  history = chat.messages;
  wipeMessages();
  for (const msg of history) {
    const display = msg.role === 'user'
      ? msg.content.replace(/ \/no_think$/, '')
      : msg.content;
    appendBubble(msg.role, toHtml(display));
  }

  const { repo, file } = chat.model;
  if ($repoInput.value.trim() !== repo) {
    $repoInput.value = repo;
    await detectGGUFFiles(file);
  } else if ($selWrap.style.display !== 'none') {
    selectModelFile(file);
  } else {
    await detectGGUFFiles(file);
  }

  setStatus(
    `Viewing saved chat · model: <b>${esc(file)}</b> — click <b>Load Model</b> to continue.`,
    'ok'
  );
}

function selectModelFile(file) {
  for (const opt of $sel.options) {
    if (opt.value === file) { $sel.value = file; return true; }
  }
  return false;
}

// ─── Cache presence check ─────────────────────────────────────────
async function loadCachedFiles() {
  try {
    const tmp = new Wllama(CONFIG_PATHS);
    const entries = await tmp.cacheManager.list();
    // name format: {hash}_{filename}.gguf — strip the hash prefix
    cachedFiles = new Set(
      entries.map(e => e.name?.replace(/^[0-9a-f]+_/, '') ?? '').filter(Boolean)
    );
    console.log('[cache] cached filenames:', [...cachedFiles]);
  } catch (err) {
    console.error('[cache] loadCachedFiles failed:', err);
    cachedFiles = new Set();
  }
}

function markCachedOptions() {
  for (const opt of $sel.options) {
    opt.textContent = opt.textContent.replace(/^✓ /, '');
    if (cachedFiles.has(opt.value)) opt.textContent = '✓ ' + opt.textContent;
  }
}

// ─── Detect GGUF files from HF repo ───────────────────────────────
async function detectGGUFFiles(preselectFile = null) {
  const repo = $repoInput.value.trim();
  if (!repo) {
    setStatus('⚠️ Enter a HuggingFace model ID (e.g., meta-llama/Llama-2-7b-hf)', 'busy');
    return;
  }

  $detectBtn.disabled = true;
  setStatus(`⏳ Scanning ${repo} for GGUF files…`, 'busy');

  try {
    const response = await fetch(`https://huggingface.co/api/models/${repo}`);
    if (!response.ok) throw new Error('Model not found or not accessible');

    const modelData = await response.json();
    const ggufFiles = (modelData.siblings || []).filter(f => f.rfilename.endsWith('.gguf'));

    if (ggufFiles.length === 0) {
      setStatus(
        `✗ No GGUF files found in ${repo}. ` +
        `<span style="font-size:11px;display:block;margin-top:2px">💡 Use a repo with pre-converted GGUF files.</span>`,
        'err'
      );
      $selWrap.style.display = 'none';
      $loadBtn.disabled = true;
      return;
    }

    HF_REPO = repo;

    // Sizes from tree API
    const sizeMap = new Map();
    try {
      const treeRes = await fetch(`https://huggingface.co/api/models/${repo}/tree/main`);
      if (treeRes.ok) {
        (await treeRes.json()).forEach(f => { if (f.size) sizeMap.set(f.path, f.size); });
      }
    } catch (_) {}

    // Group shards
    const models = new Map();
    ggufFiles.forEach(f => {
      const match    = f.rfilename.match(/^(.+?)-\d+-of-\d+\.gguf$/);
      const baseName = match ? match[1] : f.rfilename.replace('.gguf', '');
      if (!models.has(baseName)) models.set(baseName, []);
      models.get(baseName).push(f.rfilename);
    });

    $sel.innerHTML = '';
    Array.from(models.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .forEach(([baseName, files]) => {
        const totalBytes = files.reduce((s, f) => s + (sizeMap.get(f) || 0), 0);
        const shardInfo  = files.length > 1 ? ` (${files.length} shards)` : '';
        const sizeLabel  = totalBytes > 0 ? ` ~${(totalBytes / 1024 ** 3).toFixed(2)} GB` : '';
        const opt        = document.createElement('option');
        opt.value        = files[0];
        opt.textContent  = `${baseName}${shardInfo}${sizeLabel}`;
        $sel.appendChild(opt);
      });

    $selWrap.style.display = 'block';
    $loadBtn.disabled = false;
    markCachedOptions();
    if (preselectFile) selectModelFile(preselectFile);

    setStatus(`✓ Found ${ggufFiles.length} GGUF file(s) — select a model and click Load`, 'ok');

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
  if (!file) { setStatus('Select a model first', 'err'); return; }

  [$loadBtn, $detectBtn, $sel].forEach(el => { el.disabled = true; });
  [$gpuBtn, $gpuBtnM].forEach($b => { $b.disabled = true; });

  if (wllama) {
    try { await wllama.exit(); } catch (_) {}
    wllama = null; isLoaded = false;
  }

  if (activeChatId === null) { history = []; wipeMessages(); }

  wllama = new Wllama(CONFIG_PATHS, { parallelDownloads: 5 });

  if (GPU_SUPPORTED && gpuEnabled && IS_FIREFOX) {
    wllama.setCompat('default', 'firefox_safari');
  } else if (GPU_SUPPORTED && gpuEnabled) {
    wllama.setCompat('default');
  }

  const backendLabel = !GPU_SUPPORTED
    ? '🧵 CPU'
    : gpuEnabled ? (IS_FIREFOX ? '🟡 WebGPU (compat)' : '🚀 WebGPU') : '🧵 CPU (GPU off)';

  setStatus(`⌛ Downloading <b>${file}</b>… <span style="opacity:.6">${backendLabel}</span>`, 'busy');
  setProgress(1);

  try {
    await wllama.loadModelFromHF(
      { repo: HF_REPO, file },
      {
        n_ctx: 4096,
        n_gpu_layers: GPU_SUPPORTED && gpuEnabled ? 99999 : 0,
        progressCallback({ loaded, total }) {
          const pct      = total > 0 ? Math.round(loaded / total * 100) : 0;
          const loadedGB = (loaded / 1024 ** 3).toFixed(2);
          const totalGB  = total > 0 ? (total / 1024 ** 3).toFixed(2) : '?';
          const byteInfo = total > 0 ? ` (${loadedGB} GB / ${totalGB} GB)` : '';
          setProgress(pct);
          setStatus(
            `⌛ Downloading <b>${file}</b>… ${pct}%${byteInfo} <span style="opacity:.6">${backendLabel}</span>`,
            'busy'
          );
        },
      }
    );

    setProgress(0);
    isLoaded = true;
    $input.disabled     = false;
    $sendBtn.disabled   = false;
    [$clearChat, $clearChatM].forEach($b => { $b.disabled = false; });
    $loadBtn.textContent = '↺ Reload';
    setStatus(`✓ <b>${file}</b> ready · ${backendLabel}`, 'ok');

    await loadCachedFiles();
    markCachedOptions();

    console.group('[wllama] Model load result');
    console.log('WebGPU supported:', GPU_SUPPORTED);
    console.log('WebGPU enabled:', GPU_SUPPORTED && gpuEnabled);
    console.log('Backend:', backendLabel);
    console.log('n_gpu_layers:', GPU_SUPPORTED && gpuEnabled ? 99999 : 0);
    console.log('File:', file, '| Repo:', HF_REPO);
    try {
      if (typeof wllama.getModelMetadata === 'function')
        console.log('Metadata:', await wllama.getModelMetadata());
    } catch (_) {}
    console.groupEnd();

    $input.focus();

  } catch (err) {
    setProgress(0);
    setStatus(`✗ Load failed: ${esc(err.message)}`, 'err');
  } finally {
    [$loadBtn, $detectBtn, $sel].forEach(el => { el.disabled = false; });
    [$gpuBtn, $gpuBtnM].forEach($b => { $b.disabled = false; });
  }
}

// ─── GPU toggle ───────────────────────────────────────────────────
function toggleGpu() {
  if (isGenerating) return;
  gpuEnabled = !gpuEnabled;
  syncGpuBtn();
  if (isLoaded)
    setStatus(`⚠️ Backend changed to <b>${gpuEnabled ? 'WebGPU' : 'CPU'}</b> — click <b>↺ Reload</b> to apply.`, 'busy');
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
  let reply   = '';

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
    persistCurrentChat();

  } catch (err) {
    const aborted = err.name === 'AbortError' || abortCtrl.signal.aborted;
    if (aborted) {
      $body.innerHTML =
        (reply ? toHtml(reply) : '<em style="opacity:.5">…</em>') +
        '<br><em style="color:var(--muted);font-size:12px">(stopped)</em>';
      if (reply) { history.push({ role: 'assistant', content: reply }); persistCurrentChat(); }
    } else {
      $body.innerHTML = `<em style="color:#ef4444">Error: ${esc(err.message)}</em>`;
    }
  } finally {
    setGenerating(false);
    $input.focus();
  }
}

// ─── New chat ─────────────────────────────────────────────────────
function newChat() {
  if (isGenerating) return;
  activeChatId = null;
  history      = [];
  wipeMessages();
  renderChatList();
  setStatus('New chat started.');
  if (isLoaded) $input.focus();
}

// ─── Clear cache ──────────────────────────────────────────────────
async function clearCache() {
  if (!confirm('Delete all cached model files from OPFS?\nThey will re-download on next use.')) return;

  [$clearCache, $clearCacheM].forEach($b => { $b.disabled = true; });

  if (wllama && isLoaded) {
    try { await wllama.exit(); } catch (_) {}
    wllama   = null;
    isLoaded = false;
    $input.disabled     = true;
    $sendBtn.disabled   = true;
    [$clearChat, $clearChatM].forEach($b => { $b.disabled = true; });
    $loadBtn.textContent = 'Load Model';
  }

  try {
    const root = await navigator.storage.getDirectory();
    for await (const name of root.keys()) {
      try { await root.removeEntry(name, { recursive: true }); } catch (_) {}
    }
    if ('caches' in window)
      for (const key of await caches.keys()) await caches.delete(key);
    if ('indexedDB' in window) {
      const dbs = await indexedDB.databases();
      await Promise.all(
        dbs.filter(db => db.name?.match(/wllama|localforage/i))
           .map(db => new Promise(res => {
             const req = indexedDB.deleteDatabase(db.name);
             req.onsuccess = req.onerror = res;
           }))
      );
    }

    history = [];
    wipeMessages();
    cachedFiles = new Set();
    markCachedOptions();
    setStatus('🗑 Cache cleared — click <b>Load Model</b> to re-download.', 'ok');
  } catch (err) {
    setStatus(`✗ Cache clear failed: ${esc(err.message)}`, 'err');
  } finally {
    [$clearCache, $clearCacheM].forEach($b => { $b.disabled = false; });
  }
}

// ─── Textarea auto-resize ─────────────────────────────────────────
function resizeInput() {
  $input.style.height = 'auto';
  $input.style.height = Math.min($input.scrollHeight, 144) + 'px';
}

// ─── Event wiring ─────────────────────────────────────────────────
$detectBtn .addEventListener('click',   () => detectGGUFFiles());
$repoInput .addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); detectGGUFFiles(); } });
$loadBtn   .addEventListener('click',   loadModel);
[$gpuBtn, $gpuBtnM]           .forEach($b => $b.addEventListener('click', toggleGpu));
[$stopBtn, $stopBtnM]         .forEach($b => $b.addEventListener('click', () => abortCtrl?.abort()));
[$clearChat, $clearChatM]     .forEach($b => $b.addEventListener('click', newChat));
[$clearCache, $clearCacheM]   .forEach($b => $b.addEventListener('click', clearCache));
$newChatBtn.addEventListener('click',   newChat);
$sendBtn   .addEventListener('click',   sendMessage);
$input     .addEventListener('input',   resizeInput);
$input     .addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

// ─── Initialize ───────────────────────────────────────────────────
$repoInput.value = HF_REPO;
renderChatList();
loadCachedFiles().then(() => detectGGUFFiles());