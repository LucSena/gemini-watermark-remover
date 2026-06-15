import {
    WatermarkEngine,
    detectWatermarkConfig,
    calculateWatermarkPosition
} from './core/watermarkEngine.js';
import { WatermarkWorkerClient, canUseWatermarkWorker } from './core/workerClient.js';
import {
    isConfirmedWatermarkDecision,
    resolveDisplayWatermarkInfo
} from './core/watermarkDisplay.js';
import { canvasToBlob } from './core/canvasBlob.js';
import i18n from './i18n.js';
import {
    loadImage,
    setStatusMessage,
    showLoading,
    hideLoading
} from './utils.js';
import JSZip from 'jszip';
import mediumZoom from 'medium-zoom';
import { removeBackground } from '@imgly/background-removal';
import { ImageSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';

// ── Global state ─────────────────────────────────────────────────────────────

let enginePromise = null;
let workerClient = null;
let imageQueue = [];
let processedCount = 0;
let zoom = null;
// 'watermark' | 'portrait' | 'general'
let currentMode = 'watermark';

// ── DOM refs ──────────────────────────────────────────────────────────────────

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const singlePreview = document.getElementById('singlePreview');
const multiPreview = document.getElementById('multiPreview');
const imageList = document.getElementById('imageList');
const progressText = document.getElementById('progressText');
const downloadAllBtn = document.getElementById('downloadAllBtn');
const originalImage = document.getElementById('originalImage');
const processedImage = document.getElementById('processedImage');
const originalInfo = document.getElementById('originalInfo');
const processedInfo = document.getElementById('processedInfo');
const downloadBtn = document.getElementById('downloadBtn');
const copyBtn = document.getElementById('copyBtn');
const resetBtn = document.getElementById('resetBtn');

// ── Watermark engine ──────────────────────────────────────────────────────────

async function getEngine() {
    if (!enginePromise) {
        enginePromise = WatermarkEngine.create().catch((error) => {
            enginePromise = null;
            throw error;
        });
    }
    return enginePromise;
}

function getEstimatedWatermarkInfo(item) {
    if (!item?.originalImg) return null;
    const { width, height } = item.originalImg;
    const config = detectWatermarkConfig(width, height);
    const position = calculateWatermarkPosition(width, height, config);
    return { size: config.logoSize, position, config };
}

function disableWorkerClient(reason) {
    if (!workerClient) return;
    console.warn('disable worker path, fallback to main thread:', reason);
    workerClient.dispose();
    workerClient = null;
}

async function processImageWithBestPath(file, fallbackImage, options = {}) {
    if (workerClient) {
        try {
            return await workerClient.processBlob(file, options);
        } catch (error) {
            console.warn('worker process failed, fallback to main thread:', error);
            disableWorkerClient(error);
        }
    }
    const engine = await getEngine();
    const canvas = await engine.removeWatermarkFromImage(fallbackImage, options);
    const blob = await canvasToBlob(canvas);
    return { blob, meta: canvas.__watermarkMeta || null };
}

// ── Portrait mode — MediaPipe selfie segmentation ─────────────────────────────

const MEDIAPIPE_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const SELFIE_MODEL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float32/latest/selfie_segmenter.tflite';

let segmenterPromise = null;

async function getPortraitSegmenter() {
    if (!segmenterPromise) {
        segmenterPromise = (async () => {
            const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
            return ImageSegmenter.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: SELFIE_MODEL,
                    delegate: 'GPU',
                },
                outputCategoryMask: false,
                outputConfidenceMasks: true,
                runningMode: 'IMAGE',
            });
        })().catch(err => {
            segmenterPromise = null;
            throw err;
        });
    }
    return segmenterPromise;
}

async function processPortraitBgRemoval(item) {
    setStatusMessage(i18n.t('portrait.status.loading_model') + '...');
    const segmenter = await getPortraitSegmenter();
    setStatusMessage(i18n.t('portrait.status.processing') + '...');

    const img = item.originalImg;
    const result = segmenter.segment(img);

    // confidenceMasks: [0]=background, [1]=person (2-class model)
    // If model only outputs 1 mask, index 0 is foreground/person
    const masks = result.confidenceMasks;
    const personMask = masks.length > 1 ? masks[1] : masks[0];
    const maskData = personMask.getAsFloat32Array();

    // Free GPU resources before heavy canvas work
    masks.forEach(m => m.close());

    // Draw original onto canvas
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    // Apply confidence mask as alpha (smooth edges come naturally from float32 values)
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;
    for (let i = 0; i < maskData.length; i++) {
        pixels[i * 4 + 3] = Math.round(maskData[i] * 255);
    }
    ctx.putImageData(imageData, 0, 0);

    setStatusMessage('');
    return new Promise((resolve, reject) => {
        canvas.toBlob(
            blob => blob
                ? resolve({ blob, meta: { applied: true, decisionTier: 'confirmed' } })
                : reject(new Error('Failed to encode portrait result')),
            'image/png'
        );
    });
}

// ── General BG mode — @imgly/background-removal (ISNet) ──────────────────────

async function processGeneralBgRemoval(item) {
    let lastKey = '';
    const blob = await removeBackground(item.file, {
        progress: (key, current, total) => {
            if (total <= 0) return;
            const label = key !== lastKey
                ? i18n.t('bg.status.loading_model')
                : i18n.t('bg.status.processing');
            lastKey = key;
            setStatusMessage(`${label}: ${Math.round((current / total) * 100)}%`);
        },
        output: { format: 'image/png', quality: 1 },
    });
    setStatusMessage('');
    return { blob, meta: { applied: true, decisionTier: 'confirmed' } };
}

// ── Unified dispatch ──────────────────────────────────────────────────────────

async function processImage(item) {
    if (currentMode === 'portrait') return processPortraitBgRemoval(item);
    if (currentMode === 'general') return processGeneralBgRemoval(item);
    return processImageWithBestPath(item.file, item.originalImg);
}

// ── Mode switcher ─────────────────────────────────────────────────────────────

const MODE_ACTIVE = 'px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-all bg-white dark:bg-gray-700 text-gray-900 dark:text-white shadow';
const MODE_INACTIVE = 'px-4 py-2 rounded-lg text-xs sm:text-sm font-semibold transition-all text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200';

const MODE_BTN_IDS = {
    watermark: 'modeWatermark',
    portrait: 'modePortrait',
    general: 'modeGeneral',
};

function setupModeSwitcher() {
    Object.entries(MODE_BTN_IDS).forEach(([mode, id]) => {
        document.getElementById(id)?.addEventListener('click', () => setMode(mode));
    });
}

function setMode(mode) {
    currentMode = mode;

    // Update tab styles
    Object.entries(MODE_BTN_IDS).forEach(([m, id]) => {
        const btn = document.getElementById(id);
        if (btn) btn.className = m === mode ? MODE_ACTIVE : MODE_INACTIVE;
    });

    // Update subtitle
    const subtitle = document.getElementById('modeSubtitle');
    if (subtitle) {
        const key = `main.subtitle.${mode}`;
        subtitle.setAttribute('data-i18n', key);
        subtitle.textContent = i18n.t(key);
    }

    reset();
}

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
    try {
        await i18n.init();
        setupLanguageSwitch();
        setupDarkMode();
        setupModeSwitcher();
        showLoading(i18n.t('status.loading'));

        if (canUseWatermarkWorker()) {
            try {
                workerClient = new WatermarkWorkerClient({ workerUrl: './workers/watermark-worker.js' });
            } catch (workerError) {
                console.warn('worker unavailable, fallback to main thread:', workerError);
                workerClient = null;
            }
        }
        if (!workerClient) {
            getEngine().catch(e => console.warn('main thread engine warmup failed:', e));
        }

        hideLoading();
        setupEventListeners();
        setupSlider();

        zoom = mediumZoom('[data-zoomable]', {
            margin: 24,
            scrollOffset: 0,
            background: 'rgba(255, 255, 255, .6)',
        });
    } catch (error) {
        hideLoading();
        console.error('initialize error:', error);
    }
}

// ── Language switch ───────────────────────────────────────────────────────────

function setupLanguageSwitch() {
    const select = document.getElementById('langSwitch');
    if (!select) return;
    select.value = i18n.resolveLocale(i18n.locale);
    select.addEventListener('change', async () => {
        const newLocale = i18n.resolveLocale(select.value);
        if (newLocale === i18n.locale) return;
        await i18n.switchLocale(newLocale);
        select.value = i18n.locale;
        updateDynamicTexts();
    });
}

// ── Event listeners ───────────────────────────────────────────────────────────

function setupEventListeners() {
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    document.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('border-primary', 'bg-emerald-50', 'dark:bg-gray-700/50');
    });
    document.addEventListener('dragleave', (e) => {
        if (e.clientX === 0 && e.clientY === 0)
            uploadArea.classList.remove('border-primary', 'bg-emerald-50', 'dark:bg-gray-700/50');
    });
    document.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('border-primary', 'bg-emerald-50', 'dark:bg-gray-700/50');
        if (e.dataTransfer.files?.length) handleFiles(Array.from(e.dataTransfer.files));
    });
    document.addEventListener('paste', (e) => {
        const files = [];
        for (const item of e.clipboardData.items) {
            if (item.kind === 'file' && item.type.startsWith('image/')) files.push(item.getAsFile());
        }
        if (files.length) handleFiles(files);
    });

    downloadAllBtn.addEventListener('click', downloadAll);
    resetBtn.addEventListener('click', reset);
    window.addEventListener('beforeunload', () => disableWorkerClient('beforeunload'));
}

function reset() {
    singlePreview.style.display = 'none';
    multiPreview.style.display = 'none';
    imageQueue = [];
    processedCount = 0;
    fileInput.value = '';
    copyBtn.style.display = 'none';
    downloadBtn.style.display = 'none';
    setStatusMessage('');
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function handleFileSelect(e) { handleFiles(Array.from(e.target.files)); }

function handleFiles(files) {
    setStatusMessage('');
    const valid = files.filter(f => f.type.match('image/(jpeg|png|webp)') && f.size <= 20 * 1024 * 1024);
    if (!valid.length) return;

    imageQueue.forEach(item => {
        if (item.originalUrl) URL.revokeObjectURL(item.originalUrl);
        if (item.processedUrl) URL.revokeObjectURL(item.processedUrl);
    });

    imageQueue = valid.map((file, i) => ({
        id: Date.now() + i, file, name: file.name, status: 'pending',
        originalImg: null, processedMeta: null, processedBlob: null,
        originalUrl: null, processedUrl: null,
    }));
    processedCount = 0;

    if (valid.length === 1) {
        singlePreview.style.display = 'block';
        multiPreview.style.display = 'none';
        processSingle(imageQueue[0]);
    } else {
        singlePreview.style.display = 'none';
        multiPreview.style.display = 'block';
        imageList.innerHTML = '';
        updateProgress();
        multiPreview.scrollIntoView({ behavior: 'smooth', block: 'start' });
        imageQueue.forEach(item => createImageCard(item));
        processQueue();
    }
}

// ── Meta rendering ────────────────────────────────────────────────────────────

function isBgMode() { return currentMode === 'portrait' || currentMode === 'general'; }

function renderSingleImageMeta(item) {
    if (!item?.originalImg) return;
    if (isBgMode()) {
        originalInfo.innerHTML = `<p>${i18n.t('info.size')}: ${item.originalImg.width}×${item.originalImg.height}</p>`;
        return;
    }
    const watermarkInfo = resolveDisplayWatermarkInfo(item, getEstimatedWatermarkInfo(item));
    if (!watermarkInfo) return;
    originalInfo.innerHTML = `
        <p>${i18n.t('info.size')}: ${item.originalImg.width}×${item.originalImg.height}</p>
        <p>${i18n.t('info.watermark')}: ${watermarkInfo.size}×${watermarkInfo.size}</p>
        <p>${i18n.t('info.position')}: (${watermarkInfo.position.x},${watermarkInfo.position.y})</p>
    `;
}

function getProcessedStatusLabel(item) {
    if (isBgMode()) return i18n.t('info.bg.removed');
    return !isConfirmedWatermarkDecision(item) ? i18n.t('info.skipped') : i18n.t('info.removed');
}

function renderSingleProcessedMeta(item) {
    if (!item?.originalImg) return;
    if (isBgMode()) {
        processedInfo.innerHTML = `
            <p>${i18n.t('info.size')}: ${item.originalImg.width}×${item.originalImg.height}</p>
            <p>${i18n.t('info.status')}: ${i18n.t('info.bg.removed')}</p>
        `;
        return;
    }
    const watermarkInfo = resolveDisplayWatermarkInfo(item, getEstimatedWatermarkInfo(item));
    const showInfo = watermarkInfo && isConfirmedWatermarkDecision(item);
    processedInfo.innerHTML = `
        <p>${i18n.t('info.size')}: ${item.originalImg.width}×${item.originalImg.height}</p>
        ${showInfo ? `<p>${i18n.t('info.watermark')}: ${watermarkInfo.size}×${watermarkInfo.size}</p>` : ''}
        ${showInfo ? `<p>${i18n.t('info.position')}: (${watermarkInfo.position.x},${watermarkInfo.position.y})</p>` : ''}
        <p>${i18n.t('info.status')}: ${getProcessedStatusLabel(item)}</p>
    `;
}

// ── Single image processing ───────────────────────────────────────────────────

async function processSingle(item) {
    try {
        const img = await loadImage(item.file);
        item.originalImg = img;
        originalImage.src = img.src;
        renderSingleImageMeta(item);

        const processed = await processImage(item);
        item.processedMeta = processed.meta;
        item.processedBlob = processed.blob;

        renderSingleImageMeta(item);
        item.processedUrl = URL.createObjectURL(processed.blob);
        processedImage.src = item.processedUrl;

        document.getElementById('processedOverlay').style.display = 'block';
        document.getElementById('sliderHandle').style.display = 'flex';
        processedInfo.style.display = 'block';

        copyBtn.style.display = 'flex';
        copyBtn.onclick = () => copyImage(item);
        downloadBtn.style.display = 'flex';
        downloadBtn.onclick = () => downloadImage(item);

        renderSingleProcessedMeta(item);
        document.getElementById('comparisonContainer').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
        console.error(error);
        setStatusMessage(i18n.t('status.failed'), 'warn');
    }
}

// ── Batch processing ──────────────────────────────────────────────────────────

function renderImageCardStatus(item) {
    if (!item) return;
    if (item.status === 'pending') { updateStatus(item.id, i18n.t('status.pending')); return; }
    if (item.status === 'processing') { updateStatus(item.id, i18n.t('status.processing')); return; }
    if (item.status === 'error') { updateStatus(item.id, i18n.t('status.failed')); return; }
    if (item.status !== 'completed' || !item.originalImg) return;

    if (isBgMode()) {
        updateStatus(item.id, `
            <p>${i18n.t('info.size')}: ${item.originalImg.width}×${item.originalImg.height}</p>
            <p>${i18n.t('info.status')}: ${i18n.t('info.bg.removed')}</p>
        `, true);
        return;
    }
    const watermarkInfo = resolveDisplayWatermarkInfo(item, getEstimatedWatermarkInfo(item));
    const showInfo = watermarkInfo && isConfirmedWatermarkDecision(item);
    let html = `<p>${i18n.t('info.size')}: ${item.originalImg.width}×${item.originalImg.height}</p>`;
    if (showInfo) {
        html += `<p>${i18n.t('info.watermark')}: ${watermarkInfo.size}×${watermarkInfo.size}</p>
        <p>${i18n.t('info.position')}: (${watermarkInfo.position.x},${watermarkInfo.position.y})</p>`;
    }
    html += `<p>${i18n.t('info.status')}: ${getProcessedStatusLabel(item)}</p>`;
    updateStatus(item.id, html, true);
}

function createImageCard(item) {
    const card = document.createElement('div');
    card.id = `card-${item.id}`;
    card.className = 'bg-white md:h-[140px] rounded-xl shadow-card border border-gray-100 overflow-hidden';
    card.innerHTML = `
        <div class="flex flex-wrap h-full">
            <div class="w-full md:w-auto h-full flex border-b border-gray-100">
                <div class="w-24 md:w-48 flex-shrink-0 bg-gray-50 p-2 flex items-center justify-center">
                    <img id="result-${item.id}" class="max-w-full max-h-24 md:max-h-full rounded" data-zoomable />
                </div>
                <div class="flex-1 p-4 flex flex-col min-w-0">
                    <h4 class="image-name font-semibold text-sm text-gray-900 mb-2 truncate"></h4>
                    <div class="text-xs text-gray-500" id="status-${item.id}">${i18n.t('status.pending')}</div>
                </div>
            </div>
            <div class="w-full md:w-auto ml-auto flex-shrink-0 p-2 md:p-4 flex flex-col md:flex-row items-center justify-center gap-2">
                <button id="copy-${item.id}" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs md:text-sm hidden flex items-center gap-1">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m-1 10H8m4-3H8m1.5 6H8"></path></svg>
                    <span data-i18n="btn.copy">${i18n.t('btn.copy')}</span>
                </button>
                <button id="download-${item.id}" class="px-4 py-2 bg-gray-900 hover:bg-gray-800 text-white rounded-lg text-xs md:text-sm hidden">
                    <span data-i18n="btn.download">${i18n.t('btn.download')}</span>
                </button>
            </div>
        </div>
    `;
    imageList.appendChild(card);
    const nameEl = card.querySelector('.image-name');
    if (nameEl) {
        nameEl.textContent = String(item.name || '');
        nameEl.title = nameEl.textContent;
    }
}

async function processQueue() {
    await Promise.all(imageQueue.map(async item => {
        const img = await loadImage(item.file);
        item.originalImg = img;
        item.originalUrl = img.src;
        document.getElementById(`result-${item.id}`).src = img.src;
        zoom.attach(`#result-${item.id}`);
    }));

    // AI bg removal: 1 at a time to avoid OOM; watermark: 3 concurrent
    const concurrency = isBgMode() ? 1 : 3;
    for (let i = 0; i < imageQueue.length; i += concurrency) {
        await Promise.all(imageQueue.slice(i, i + concurrency).map(async item => {
            if (item.status !== 'pending') return;
            item.status = 'processing';
            renderImageCardStatus(item);
            try {
                const processed = await processImage(item);
                item.processedMeta = processed.meta;
                item.processedBlob = processed.blob;
                item.processedUrl = URL.createObjectURL(processed.blob);
                document.getElementById(`result-${item.id}`).src = item.processedUrl;
                item.status = 'completed';
                renderImageCardStatus(item);

                const copyEl = document.getElementById(`copy-${item.id}`);
                copyEl.classList.remove('hidden');
                copyEl.onclick = () => copyImage(item, copyEl);

                const dlEl = document.getElementById(`download-${item.id}`);
                dlEl.classList.remove('hidden');
                dlEl.onclick = () => downloadImage(item);

                processedCount++;
                updateProgress();
            } catch (error) {
                item.status = 'error';
                renderImageCardStatus(item);
                console.error(error);
            }
        }));
    }
    if (processedCount > 0) downloadAllBtn.style.display = 'flex';
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function updateStatus(id, text, isHtml = false) {
    const el = document.getElementById(`status-${id}`);
    if (el) el.innerHTML = isHtml ? text : text.replace(/\n/g, '<br>');
}

function updateProgress() {
    progressText.textContent = `${i18n.t('progress.text')}: ${processedCount}/${imageQueue.length}`;
}

function updateDynamicTexts() {
    // Re-translate mode buttons
    Object.entries(MODE_BTN_IDS).forEach(([mode, id]) => {
        const btn = document.getElementById(id);
        if (btn) btn.textContent = i18n.t(`mode.${mode}`);
    });

    // Re-translate subtitle
    const subtitle = document.getElementById('modeSubtitle');
    if (subtitle) subtitle.textContent = i18n.t(subtitle.getAttribute('data-i18n') || `main.subtitle.${currentMode}`);

    if (imageQueue.length > 0) {
        updateProgress();
        imageQueue.forEach(item => renderImageCardStatus(item));
    }
    if (singlePreview.style.display !== 'none' && imageQueue.length === 1) {
        const [item] = imageQueue;
        renderSingleImageMeta(item);
        if (item?.processedBlob) renderSingleProcessedMeta(item);
    }
}

function getFilePrefix() {
    if (currentMode === 'portrait') return 'portrait_nobg_';
    if (currentMode === 'general') return 'nobg_';
    return 'unwatermarked_';
}

async function copyImage(item, targetBtn = copyBtn) {
    if (!navigator.clipboard || !window.ClipboardItem) {
        setStatusMessage(i18n.t('status.unsupported'), 'warn');
        return;
    }
    try {
        if (!item.processedBlob) return;
        await navigator.clipboard.write([new ClipboardItem({ [item.processedBlob.type]: item.processedBlob })]);
        const span = targetBtn.querySelector('span');
        const svg = targetBtn.querySelector('svg');
        const origSvg = svg?.innerHTML;
        span.textContent = i18n.t('status.copied');
        if (svg) svg.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>';
        setTimeout(() => {
            span.textContent = i18n.t('btn.copy');
            if (svg && origSvg) svg.innerHTML = origSvg;
        }, 2000);
    } catch (err) {
        console.error('Failed to copy image:', err);
        setStatusMessage(i18n.t('status.copy_failed'), 'warn');
    }
}

function downloadImage(item) {
    const a = document.createElement('a');
    a.href = item.processedUrl;
    a.download = `${getFilePrefix()}${item.name.replace(/\.[^.]+$/, '')}.png`;
    a.click();
}

async function downloadAll() {
    const completed = imageQueue.filter(item => item.status === 'completed');
    if (!completed.length) return;
    const zip = new JSZip();
    const prefix = getFilePrefix();
    completed.forEach(item => zip.file(`${prefix}${item.name.replace(/\.[^.]+$/, '')}.png`, item.processedBlob));
    const blob = await zip.generateAsync({ type: 'blob' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${prefix}${Date.now()}.zip`;
    a.click();
}

// ── UI setup ──────────────────────────────────────────────────────────────────

function setupDarkMode() {
    const themeToggle = document.getElementById('themeToggle');
    const html = document.documentElement;
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        html.classList.add('dark');
    }
    themeToggle.addEventListener('click', () => {
        html.classList.toggle('dark');
        localStorage.theme = html.classList.contains('dark') ? 'dark' : 'light';
    });
}

function setupSlider() {
    const container = document.getElementById('comparisonContainer');
    const overlay = document.getElementById('processedOverlay');
    const handle = document.getElementById('sliderHandle');
    let isDown = false;

    function move(e) {
        if (!isDown) return;
        const rect = container.getBoundingClientRect();
        const clientX = e.clientX ?? e.touches?.[0]?.clientX;
        if (clientX == null) return;
        const percent = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1) * 100;
        overlay.style.width = `${percent}%`;
        handle.style.left = `${percent}%`;
    }

    container.addEventListener('mousedown', e => { isDown = true; move(e); });
    window.addEventListener('mouseup', () => { isDown = false; });
    window.addEventListener('mousemove', move);
    container.addEventListener('touchstart', e => { isDown = true; move(e); });
    window.addEventListener('touchend', () => { isDown = false; });
    window.addEventListener('touchmove', move);
}

init();
