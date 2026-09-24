from __future__ import annotations

import numpy as np

from temporal_evidence_snn.data.static_images import as_image_batch
from temporal_evidence_snn.frontend import (
    event_surface_frontend_from_config,
    StaticLatencyFrontend,
    StaticLatencyFrontendConfig,
    latency_to_spike_bins,
    static_latency_frontend_from_config,
)


def test_as_image_batch_accepts_grayscale_and_rgb_layouts() -> None:
    gray = np.arange(28 * 28, dtype=np.uint8).reshape(28, 28)
    gray_stack = np.zeros((2, 8, 8), dtype=np.uint8)
    event_surface = np.zeros((2, 10, 8, 8), dtype=np.float32)
    rgb = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 255

    gray_batch = as_image_batch(gray, channels=1)
    gray_stack_batch = as_image_batch(gray_stack, channels=1)
    event_surface_batch = as_image_batch(event_surface, channels=10)
    rgb_batch = as_image_batch(rgb, channels=3)

    assert gray_batch.shape == (1, 1, 28, 28)
    assert gray_stack_batch.shape == (2, 1, 8, 8)
    assert event_surface_batch.shape == (2, 10, 8, 8)
    assert rgb_batch.shape == (2, 3, 32, 32)
    assert gray_batch.dtype == np.float32
    assert rgb_batch.max() == 1.0


def test_static_latency_frontend_outputs_calibrated_on_off_latency_maps() -> None:
    images = np.zeros((3, 1, 8, 8), dtype=np.float32)
    images[0, 0, 2:6, 2:6] = 1.0
    images[1, 0, 1:7, 3:5] = 0.8
    images[2, 0, 3:5, 1:7] = 0.6

    frontend = StaticLatencyFrontend(
        StaticLatencyFrontendConfig(
            input_channels=1,
            split_polarities=True,
            whitening_patch_size=3,
            whitening_stride=1,
            whitening_max_samples=512,
            fit_batch_images=3,
            response_floor=0.05,
            time_steps=11,
        )
    ).fit(images)

    latencies = frontend.transform_latency(images)

    assert latencies.shape == (3, 2, 8, 8)
    finite = latencies[np.isfinite(latencies)]
    assert finite.size > 0
    assert float(finite.min()) >= 0.0
    assert float(finite.max()) <= 1.0
    assert np.isinf(latencies).any()


def test_latency_to_spike_bins_has_one_spike_per_active_cell() -> None:
    latencies = np.array(
        [
            [
                [[0.0, 0.5], [1.0, np.inf]],
            ],
        ],
        dtype=np.float32,
    )

    spikes = latency_to_spike_bins(latencies, time_steps=5)

    assert spikes.shape == (5, 1, 1, 2, 2)
    assert spikes[:, 0, 0, 0, 0].argmax() == 0
    assert spikes[:, 0, 0, 0, 1].argmax() == 2
    assert spikes[:, 0, 0, 1, 0].argmax() == 4
    assert spikes[:, 0, 0, 1, 1].sum() == 0.0
    assert spikes.sum() == 3.0


def test_static_latency_frontend_builder_reads_config() -> None:
    frontend = static_latency_frontend_from_config(
        {
            "input_type": "rgb_image",
            "time_steps": 7,
            "local_whitening": {"patch_size": 5, "stride": 1, "eps": 1.0e-3},
            "polarity_split": {"enabled": True},
            "calibration": {"mode": "dataset_feature", "oracle_compat": True},
            "response": {"floor": 0.01, "gamma": 1.0},
        }
    )

    assert frontend.config.input_channels == 3
    assert frontend.config.time_steps == 7
    assert frontend.config.whitening_patch_size == 5


def test_event_stream_frontend_accepts_multichannel_event_surfaces() -> None:
    images = np.zeros((2, 10, 8, 8), dtype=np.float32)
    images[:, :, 2:6, 2:6] = 1.0
    frontend = event_surface_frontend_from_config(
        {
            "input_type": "event_stream",
            "output_channels": 10,
            "time_steps": 9,
        }
    )

    latency = frontend.fit(images).transform_latency(images)

    assert frontend.config.input_channels == 10
    assert frontend.output_channels == 10
    assert latency.shape == (2, 10, 8, 8)
