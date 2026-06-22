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
const $stopBtn        = document.getElementById('stop-btn');
const $clearChat      = document.getElementById('clear-chat-btn');
const $clearCache     = document.getElementById('clear-cache-btn');
const $chat           = document.getElementById('chat');
const $empty          = document.getElementById('empty');
const $input          = document.getElementById('user-input');
const $sendBtn        = document.getElementById('send-btn');
const $status         = document.getElementById('status');
const $progWrap       = document.getElementById('prog-wrap');
const $progBar        = document.getElementById('prog-bar');
const $sidebar        = document.getElementById('sidebar');
const $chatList       = document.getElementById('chat-list');
const $sidebarToggle  = document.getElementById('sidebar-toggle');
const $newChatBtn     = document.getElementById('new-chat-btn');

// ─── App state ────────────────────────────────────────────────────
let wllama       = null;
let isLoaded     = false;
let isGenerating = false;
let history      = [];
let abortCtrl    = null;
let cachedFiles  = new Set(); // filenames known to be in OPFS cache

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
        // adapter.info is the current API; requestAdapterInfo() was removed in Chrome 121+
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

// ─── Saved chats (localStorage) ───────────────────────────────────
// Schema: { id, title, model: { repo, file }, messages: [{role,content}], updatedAt }

const STORAGE_KEY = 'wllama_chats';

let activeChatId = null; // id of the currently open saved chat, or null for unsaved

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
      '<li class="px-3 py-3 text-[12px] text-muted italic">No saved chats yet.</li>';
    return;
  }

  chats.forEach(chat => {
    const isActive = chat.id === activeChatId;
    const li = document.createElement('li');
    li.className =
      `group flex items-start gap-1.5 px-2.5 py-2 cursor-pointer rounded-md mx-1 my-0.5
       hover:bg-surface2 transition-colors ${isActive ? 'bg-surface2 ring-1 ring-violet-700/40' : ''}`;

    li.innerHTML = `
      <div class="flex-1 min-w-0">
        <div class="text-[12.5px] font-medium truncate leading-snug">${esc(chat.title)}</div>
        <div class="text-[11px] text-muted truncate mt-0.5" title="${esc(chat.model.file)}">${esc(chat.model.file)}</div>
      </div>
      <button class="del-btn shrink-0 opacity-0 group-hover:opacity-100 text-muted hover:text-red-400
                     transition-opacity mt-0.5 leading-none text-[14px]" title="Delete chat">×</button>`;

    li.addEventListener('click', () => openSavedChat(chat.id));
    li.querySelector('.del-btn').addEventListener('click', e => {
      e.stopPropagation();
      if (confirm('Delete this chat?')) deleteChat(chat.id);
    });

    $chatList.appendChild(li);
  });
}

// Save current in-progress chat (called after each assistant reply)
function persistCurrentChat() {
  if (history.length === 0) return;
  if (!isLoaded) return;

  const file = $sel.value;
  const repo = HF_REPO;
  if (!file) return;

  // derive title from first user message
  const firstUser = history.find(m => m.role === 'user');
  const rawTitle  = firstUser ? firstUser.content.replace(/ \/no_think$/, '') : 'Chat';
  const title     = rawTitle.length > 48 ? rawTitle.slice(0, 48) + '…' : rawTitle;

  if (!activeChatId) activeChatId = `chat_${Date.now()}`;

  upsertChat(activeChatId, {
    title,
    model: { repo, file },
    messages: history,
  });
}

// Restore a saved chat into the UI (read-only replay — no model needed)
async function openSavedChat(id) {
  if (isGenerating) return;

  const chats = loadAllChats();
  const chat  = chats.find(c => c.id === id);
  if (!chat) return;

  // Switch active id and repaint list
  activeChatId = id;
  renderChatList();

  // Restore history
  history = chat.messages;

  // Render messages
  wipeMessages();
  for (const msg of history) {
    // Strip /no_think suffix from displayed user messages
    const display = msg.role === 'user'
      ? msg.content.replace(/ \/no_think$/, '')
      : msg.content;
    appendBubble(msg.role, toHtml(display));
  }

  // Pre-select the model in the toolbar
  const { repo, file } = chat.model;
  if ($repoInput.value.trim() !== repo) {
    $repoInput.value = repo;
    // Re-detect so the select is populated, then pick the right file
    await detectGGUFFiles(file);
  } else if ($selWrap.style.display !== 'none') {
    // Select already populated — just pick the right option
    selectModelFile(file);
  } else {
    await detectGGUFFiles(file);
  }

  setStatus(
    `Viewing saved chat · model: <b>${esc(file)}</b> — click <b>Load Model</b> to continue chatting.`,
    'ok'
  );
}

// Helper: select a specific file in the dropdown (if present)
function selectModelFile(file) {
  for (const opt of $sel.options) {
    if (opt.value === file) { $sel.value = file; return true; }
  }
  return false;
}

// ─── Sidebar toggle ───────────────────────────────────────────────
$sidebarToggle.addEventListener('click', () => {
  const hidden = $sidebar.classList.toggle('hidden');
  $sidebarToggle.setAttribute('aria-pressed', String(!hidden));
});

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

// ─── Detect GGUF files from HF repo ───────────────────────────────
// Optional `preselectFile` arg: after populating the select, try to pick that file.
async function detectGGUFFiles(preselectFile = null) {
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

    // Fetch file sizes from the tree API (siblings endpoint doesn't include size)
    const sizeMap = new Map();
    try {
      const treeRes = await fetch(`https://huggingface.co/api/models/${repo}/tree/main`);
      if (treeRes.ok) {
        const treeData = await treeRes.json();
        treeData.forEach(f => { if (f.size) sizeMap.set(f.path, f.size); });
      }
    } catch (_) { /* size labels will be omitted if this fails */ }

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
      const totalBytes = files.reduce((sum, f) => sum + (sizeMap.get(f) || 0), 0);
      const shardInfo = files.length > 1 ? ` (${files.length} shards)` : '';
      const sizeLabel = totalBytes > 0
        ? ` ~${(totalBytes / 1024 ** 3).toFixed(2)} GB`
        : '';
      const option = document.createElement('option');
      option.value = files[0]; // first shard filename
      option.textContent = `${baseName}${shardInfo}${sizeLabel}`;
      $sel.appendChild(option);
    });

    $selWrap.style.display = 'block';
    $loadBtn.disabled = false;

    // Mark cached options with a checkmark
    markCachedOptions();

    // Pre-select a specific file if requested (e.g. when restoring a saved chat)
    if (preselectFile) selectModelFile(preselectFile);

    setStatus(`✓ Found ${ggufFiles.length} GGUF file(s) · Select a model and click Load`, 'ok');

  } catch (err) {
    setStatus(`✗ Detection failed: ${esc(err.message)}`, 'err');
    $selWrap.style.display = 'none';
    $loadBtn.disabled = true;
  } finally {
    $detectBtn.disabled = false;
  }
}

// ─── Mark cached options in the dropdown ──────────────────────────
function markCachedOptions() {
  for (const opt of $sel.options) {
    const file = opt.value;
    const isCached = cachedFiles.has(file);
    // Strip any existing prefix before re-applying
    opt.textContent = opt.textContent.replace(/^✓ /, '');
    if (isCached) opt.textContent = '✓ ' + opt.textContent;
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

  // Only wipe messages if this is a fresh load (not restoring a saved chat)
  if (activeChatId === null) {
    history = [];
    wipeMessages();
  }

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
          const loadedGB = (loaded / 1024 ** 3).toFixed(2);
          const totalGB  = total > 0 ? (total / 1024 ** 3).toFixed(2) : '?';
          const byteInfo = total > 0 ? ` (${loadedGB} GB / ${totalGB} GB)` : '';
          setProgress(pct);
          setStatus(
            `⌛ Downloading <b>${file}</b>… ${pct}%${byteInfo} <span class="opacity-60">${backendLabel}</span>`,
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

    // File is now cached — refresh the set and re-mark the dropdown
    await loadCachedFiles();
    markCachedOptions();

    console.group('[wllama] Model load result');
    console.log('WebGPU supported (isSupportWebGPU):', GPU_SUPPORTED);
    console.log('WebGPU enabled (toggle):', GPU_SUPPORTED && gpuEnabled);
    console.log('Backend label:', backendLabel);
    console.log('n_gpu_layers requested:', GPU_SUPPORTED && gpuEnabled ? 99999 : 0);
    console.log('File:', file);
    console.log('Repo:', HF_REPO);
    try {
      console.log('wllama instance keys:', Object.keys(wllama));
      if (typeof wllama.getModelMetadata === 'function') {
        const meta = await wllama.getModelMetadata();
        console.log('Model metadata:', meta);
      }
    } catch (e) {
      console.warn('Could not read wllama internals:', e);
    }
    console.groupEnd();

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
    persistCurrentChat();

  } catch (err) {
    const aborted = err.name === 'AbortError' || abortCtrl.signal.aborted;
    if (aborted) {
      $body.innerHTML =
        (reply ? toHtml(reply) : '<em class="opacity-50">…</em>') +
        '<br><em class="text-muted text-xs">(stopped)</em>';
      if (reply) {
        history.push({ role: 'assistant', content: reply });
        persistCurrentChat();
      }
    } else {
      $body.innerHTML = `<em class="text-red-400">Error: ${esc(err.message)}</em>`;
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
  history = [];
  wipeMessages();
  renderChatList();
  setStatus('New chat started.');
  if (isLoaded) $input.focus();
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

    history = [];
    wipeMessages();
    cachedFiles = new Set();
    markCachedOptions();
    setStatus('🗑 Cache cleared — click <b>Load Model</b> to re-download.', 'ok');
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
$detectBtn .addEventListener('click',   () => detectGGUFFiles());
$repoInput .addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); detectGGUFFiles(); }
});
$loadBtn   .addEventListener('click',   loadModel);
$gpuBtn    .addEventListener('click',   toggleGpu);
$stopBtn   .addEventListener('click',   () => abortCtrl?.abort());
$clearChat .addEventListener('click',   newChat);
$newChatBtn.addEventListener('click',   newChat);
$clearCache.addEventListener('click',   clearCache);
$sendBtn   .addEventListener('click',   sendMessage);
$input     .addEventListener('input',   resizeInput);
$input     .addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// ─── Initialize ───────────────────────────────────────────────────
$repoInput.value = HF_REPO;
renderChatList();
loadCachedFiles().then(() => detectGGUFFiles());