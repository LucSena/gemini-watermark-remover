import test from 'node:test';
import assert from 'node:assert/strict';

import {
    pickSupportedVideoMimeType,
    isVideoWatermarkRemovalSupported,
    VIDEO_MIME_CANDIDATES
} from '../../src/core/videoWatermarkEngine.js';

test('pickSupportedVideoMimeType returns the first supported candidate in priority order', () => {
    const isTypeSupported = (mimeType) => mimeType === VIDEO_MIME_CANDIDATES[2].mimeType
        || mimeType === VIDEO_MIME_CANDIDATES[3].mimeType;

    const result = pickSupportedVideoMimeType(VIDEO_MIME_CANDIDATES, isTypeSupported);
    assert.deepEqual(result, VIDEO_MIME_CANDIDATES[2]);
});

test('pickSupportedVideoMimeType returns null when nothing is supported', () => {
    const result = pickSupportedVideoMimeType(VIDEO_MIME_CANDIDATES, () => false);
    assert.equal(result, null);
});

test('pickSupportedVideoMimeType tolerates a predicate that throws', () => {
    const isTypeSupported = (mimeType) => {
        if (mimeType === VIDEO_MIME_CANDIDATES[0].mimeType) {
            throw new Error('unsupported mime string');
        }
        return mimeType === VIDEO_MIME_CANDIDATES[1].mimeType;
    };

    const result = pickSupportedVideoMimeType(VIDEO_MIME_CANDIDATES, isTypeSupported);
    assert.deepEqual(result, VIDEO_MIME_CANDIDATES[1]);
});

test('isVideoWatermarkRemovalSupported is false outside a browser-like environment', () => {
    assert.equal(isVideoWatermarkRemovalSupported(), false);
});
