/**
 * Video watermark removal
 *
 * Gemini's video watermark is the same fixed-position logo used on images,
 * composited identically on every frame. So instead of re-running the full
 * adaptive detection pipeline per frame (far too slow for video), we run it
 * once on a handful of sample frames to lock in {position, alphaMap,
 * alphaGain}, then apply the cheap reverse-alpha-blend (bounded to the
 * watermark's small bounding box) to every decoded frame in real time while
 * re-encoding via MediaRecorder.
 */

import { removeWatermark } from './blendModes.js';

const DEFAULT_CAPTURE_FPS = 30;
// Avoid sampling frame 0: Gemini exports sometimes fade in/out at the very
// start/end, which can hide or distort the watermark for detection.
const CALIBRATION_SAMPLE_FRACTIONS = [0.2, 0.5, 0.05, 0.8];

const VIDEO_MIME_CANDIDATES = Object.freeze([
    { mimeType: 'video/mp4;codecs=avc1.42E01E,mp4a.40.2', extension: 'mp4' },
    { mimeType: 'video/mp4', extension: 'mp4' },
    { mimeType: 'video/webm;codecs=vp9,opus', extension: 'webm' },
    { mimeType: 'video/webm;codecs=vp8,opus', extension: 'webm' },
    { mimeType: 'video/webm', extension: 'webm' }
]);

function defaultIsTypeSupported(mimeType) {
    return typeof MediaRecorder !== 'undefined' && typeof MediaRecorder.isTypeSupported === 'function'
        ? MediaRecorder.isTypeSupported(mimeType)
        : false;
}

/**
 * Pick the first supported output container/codec pair.
 * @param {Array<{mimeType: string, extension: string}>} candidates
 * @param {(mimeType: string) => boolean} isTypeSupported - injectable for testing
 */
export function pickSupportedVideoMimeType(candidates = VIDEO_MIME_CANDIDATES, isTypeSupported = defaultIsTypeSupported) {
    for (const candidate of candidates) {
        try {
            if (isTypeSupported(candidate.mimeType)) return candidate;
        } catch {
            // unsupported string can throw in some engines; just try the next one
        }
    }
    return null;
}

export function isVideoWatermarkRemovalSupported() {
    return typeof document !== 'undefined'
        && typeof HTMLCanvasElement !== 'undefined'
        && typeof HTMLCanvasElement.prototype.captureStream === 'function'
        && typeof MediaRecorder !== 'undefined';
}

function waitForEvent(target, event, { timeoutMs } = {}) {
    return new Promise((resolve, reject) => {
        let timer = null;
        const cleanup = () => {
            target.removeEventListener(event, onEvent);
            target.removeEventListener('error', onError);
            if (timer) clearTimeout(timer);
        };
        const onEvent = (e) => { cleanup(); resolve(e); };
        const onError = (e) => { cleanup(); reject(e?.target?.error || new Error(`${event} failed`)); };
        target.addEventListener(event, onEvent, { once: true });
        target.addEventListener('error', onError, { once: true });
        if (timeoutMs) {
            timer = setTimeout(() => { cleanup(); reject(new Error(`Timed out waiting for ${event}`)); }, timeoutMs);
        }
    });
}

function seekTo(video, time) {
    return new Promise((resolve, reject) => {
        const cleanup = () => {
            video.removeEventListener('seeked', onSeeked);
            video.removeEventListener('error', onError);
        };
        const onSeeked = () => { cleanup(); resolve(); };
        const onError = (e) => { cleanup(); reject(e?.target?.error || new Error('seek failed')); };
        video.addEventListener('seeked', onSeeked, { once: true });
        video.addEventListener('error', onError, { once: true });
        video.currentTime = time;
    });
}

async function calibrateVideoWatermark(engine, video) {
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    const snap = document.createElement('canvas');
    snap.width = video.videoWidth;
    snap.height = video.videoHeight;
    const ctx = snap.getContext('2d', { willReadFrequently: true });

    let lastMeta = null;
    for (const fraction of CALIBRATION_SAMPLE_FRACTIONS) {
        if (duration > 0) {
            const time = Math.min(Math.max(duration - 0.05, 0), duration * fraction);
            await seekTo(video, time);
        }

        ctx.drawImage(video, 0, 0, snap.width, snap.height);
        const resultCanvas = await engine.removeWatermarkFromImage(snap);
        const meta = resultCanvas.__watermarkMeta;
        lastMeta = meta;

        if (meta?.applied && meta.position && meta.config) {
            const alphaMap = await engine.getAlphaMap(meta.position.width);
            return {
                applied: true,
                position: meta.position,
                alphaGain: meta.alphaGain || 1,
                alphaMap,
                meta
            };
        }

        if (duration <= 0) break;
    }

    return { applied: false, meta: lastMeta };
}

function buildOutputStream(canvas, video, captureFps) {
    const tracks = [...canvas.captureStream(captureFps).getVideoTracks()];

    const captureAudio = video.captureStream || video.webkitCaptureStream;
    if (typeof captureAudio === 'function') {
        try {
            const sourceStream = captureAudio.call(video);
            tracks.push(...sourceStream.getAudioTracks());
        } catch (error) {
            console.warn('video audio capture unavailable, continuing without audio:', error);
        }
    }

    return new MediaStream(tracks);
}

/**
 * Remove the Gemini watermark from a video file, entirely client-side.
 * @param {import('./watermarkEngine.js').WatermarkEngine} engine
 * @param {File|Blob} file
 * @param {Object} [options]
 * @param {(fraction: number) => void} [options.onProgress]
 * @param {(status: 'calibrating'|'processing') => void} [options.onStatus]
 * @param {number} [options.captureFps]
 * @returns {Promise<{applied: boolean, blob?: Blob, mimeType?: string, extension?: string, meta: Object|null}>}
 */
export async function removeVideoWatermark(engine, file, options = {}) {
    const { onProgress, onStatus, captureFps = DEFAULT_CAPTURE_FPS } = options;

    if (!isVideoWatermarkRemovalSupported()) {
        const error = new Error('Video watermark removal is not supported in this browser');
        error.code = 'unsupported';
        throw error;
    }

    const mimeCandidate = pickSupportedVideoMimeType();
    if (!mimeCandidate) {
        const error = new Error('No supported video recording format in this browser');
        error.code = 'unsupported';
        throw error;
    }

    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    video.src = objectUrl;

    try {
        await waitForEvent(video, 'loadedmetadata', { timeoutMs: 30000 });
        if (!video.videoWidth || !video.videoHeight) {
            const error = new Error('Could not read video dimensions');
            error.code = 'invalid-video';
            throw error;
        }

        onStatus?.('calibrating');
        const calibration = await calibrateVideoWatermark(engine, video);
        if (!calibration.applied) {
            return { applied: false, meta: calibration.meta };
        }

        await seekTo(video, 0).catch(() => {});

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        const outputStream = buildOutputStream(canvas, video, captureFps);
        const chunks = [];
        const recorder = new MediaRecorder(outputStream, {
            mimeType: mimeCandidate.mimeType,
            videoBitsPerSecond: 8_000_000
        });
        recorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) chunks.push(e.data);
        };

        const recordingDone = new Promise((resolve, reject) => {
            recorder.onstop = resolve;
            recorder.onerror = (e) => reject(e?.error || new Error('recording failed'));
        });

        const { position, alphaMap, alphaGain } = calibration;
        const { x, y, width, height } = position;
        const localPosition = { x: 0, y: 0, width, height };

        let stopped = false;
        const supportsRvfc = typeof video.requestVideoFrameCallback === 'function';

        const drawAndClean = () => {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const region = ctx.getImageData(x, y, width, height);
            removeWatermark(region, alphaMap, localPosition, { alphaGain });
            ctx.putImageData(region, x, y);

            if (Number.isFinite(video.duration) && video.duration > 0) {
                onProgress?.(Math.min(1, video.currentTime / video.duration));
            }
        };

        let rafId = null;
        const onRvfcFrame = () => {
            if (stopped) return;
            drawAndClean();
            if (!video.paused && !video.ended) {
                video.requestVideoFrameCallback(onRvfcFrame);
            }
        };
        const rafLoop = () => {
            if (stopped) return;
            drawAndClean();
            if (!video.paused && !video.ended) {
                rafId = requestAnimationFrame(rafLoop);
            }
        };

        const finalize = () => {
            if (stopped) return;
            stopped = true;
            if (rafId !== null) cancelAnimationFrame(rafId);
            if (recorder.state !== 'inactive') recorder.stop();
        };
        video.addEventListener('ended', finalize, { once: true });

        onStatus?.('processing');
        recorder.start(250);
        if (supportsRvfc) {
            video.requestVideoFrameCallback(onRvfcFrame);
        } else {
            rafLoop();
        }
        await video.play();

        await recordingDone;
        onProgress?.(1);

        const blob = new Blob(chunks, { type: mimeCandidate.mimeType });
        return {
            applied: true,
            blob,
            mimeType: mimeCandidate.mimeType,
            extension: mimeCandidate.extension,
            meta: calibration.meta
        };
    } finally {
        video.pause();
        video.removeAttribute('src');
        video.load();
        URL.revokeObjectURL(objectUrl);
    }
}

export { VIDEO_MIME_CANDIDATES };
