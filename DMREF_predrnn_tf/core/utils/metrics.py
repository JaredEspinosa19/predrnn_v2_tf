import numpy as np


def batch_psnr(gen_frames, gt_frames):
    """
    PSNR promedio sobre un batch de frames.

    Args:
        gen_frames : np.uint8  [B, H, W] o [B, H, W, C]
        gt_frames  : np.uint8  misma forma

    Returns:
        float  PSNR promedio en dB
    """
    if gen_frames.ndim == 3:
        axis = (1, 2)
    elif gen_frames.ndim == 4:
        axis = (1, 2, 3)
    else:
        raise ValueError(f'Forma inesperada: {gen_frames.shape}')

    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse  = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return float(np.mean(psnr))
