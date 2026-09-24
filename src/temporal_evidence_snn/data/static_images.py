from __future__ import annotations

import numpy as np


def _looks_channel_first(shape: tuple[int, ...]) -> bool:
    if len(shape) != 3:
        return False
    return int(shape[0]) in {1, 2, 3, 6} and int(shape[-1]) not in {1, 2, 3, 6}


def as_image_batch(images: np.ndarray, *, channels: int | None = None) -> np.ndarray:
    """Return images as ``float32`` with shape ``(N,C,H,W)`` in ``[0,1]``.

    Accepted inputs are ``(H,W)``, ``(C,H,W)``, ``(H,W,C)``, ``(N,H,W)``,
    ``(N,C,H,W)``, and ``(N,H,W,C)``. Integer inputs are interpreted as
    8-bit image intensities. Floating inputs outside ``[0,1]`` are divided by
    their maximum finite value when that value is greater than one.
    """

    array = np.asarray(images)
    if array.ndim == 2:
        array = array[None, None, :, :]
    elif array.ndim == 3:
        if channels is not None:
            expected_channels = int(channels)
            first_axis_matches = int(array.shape[0]) == expected_channels
            last_axis_matches = int(array.shape[-1]) == expected_channels
            if first_axis_matches and not last_axis_matches:
                array = array[None, :, :, :]
            elif last_axis_matches and not first_axis_matches:
                array = np.moveaxis(array, -1, 0)[None, :, :, :]
            else:
                array = array[:, None, :, :]
        elif _looks_channel_first(tuple(array.shape)):
            array = array[None, :, :, :]
        elif int(array.shape[-1]) in {1, 2, 3, 6}:
            array = np.moveaxis(array, -1, 0)[None, :, :, :]
        else:
            array = array[:, None, :, :]
    elif array.ndim == 4:
        if channels is not None:
            expected_channels = int(channels)
            if int(array.shape[1]) == expected_channels:
                pass
            elif int(array.shape[-1]) == expected_channels:
                array = np.moveaxis(array, -1, 1)
            else:
                raise ValueError(
                    f"Cannot find expected channel axis {expected_channels} "
                    f"for image batch shape {array.shape}"
                )
        elif int(array.shape[1]) in {1, 2, 3, 6}:
            pass
        elif int(array.shape[-1]) in {1, 2, 3, 6}:
            array = np.moveaxis(array, -1, 1)
        else:
            raise ValueError(f"Cannot infer channel axis for image batch shape {array.shape}")
    else:
        raise ValueError(f"Expected image or image batch, got shape {array.shape}")

    output = array.astype(np.float32, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        output = output / 255.0
    else:
        finite = output[np.isfinite(output)]
        max_value = float(finite.max()) if finite.size else 0.0
        if max_value > 1.0:
            output = output / max_value
    output = np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)

    if channels is not None and int(output.shape[1]) != int(channels):
        if int(channels) == 1 and int(output.shape[1]) == 3:
            output = output.mean(axis=1, keepdims=True)
        elif int(channels) == 3 and int(output.shape[1]) == 1:
            output = np.repeat(output, repeats=3, axis=1)
        else:
            raise ValueError(
                f"Expected {channels} channels, got {int(output.shape[1])}"
            )
    return output
