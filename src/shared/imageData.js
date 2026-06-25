/**
 * Shared ImageData utility functions.
 * Consolidates cloneImageData to avoid duplication across core modules.
 */

export function cloneImageData(imageData) {
  if (typeof ImageData !== 'undefined' && imageData instanceof ImageData) {
    return new ImageData(
      new Uint8ClampedArray(imageData.data),
      imageData.width,
      imageData.height
    );
  }

  return {
    width: imageData.width,
    height: imageData.height,
    data: new Uint8ClampedArray(imageData.data)
  };
}
