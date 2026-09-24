from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temporal_evidence_snn.data.static_images import as_image_batch
from temporal_evidence_snn.frontend.static_latency import latency_to_spike_bins


@dataclass(frozen=True)
class EventSurfaceLatencyFrontendConfig:
    input_channels: int = 10
    time_steps: int = 120


class EventSurfaceLatencyFrontend:
    """Latency encoder for already normalized polarity-by-time-bin surfaces."""

    def __init__(self, config: EventSurfaceLatencyFrontendConfig) -> None:
        self.config = config

    @property
    def output_channels(self) -> int:
        return int(self.config.input_channels)

    def fit(self, surfaces: np.ndarray) -> "EventSurfaceLatencyFrontend":
        self._as_surfaces(surfaces)
        return self

    def _as_surfaces(self, surfaces: np.ndarray) -> np.ndarray:
        return as_image_batch(
            surfaces,
            channels=int(self.config.input_channels),
        )

    def transform_response(self, surfaces: np.ndarray) -> np.ndarray:
        return np.clip(self._as_surfaces(surfaces), 0.0, 1.0).astype(
            np.float32,
            copy=False,
        )

    def transform_latency(self, surfaces: np.ndarray) -> np.ndarray:
        response = self.transform_response(surfaces)
        latency = np.full(response.shape, np.inf, dtype=np.float32)
        active = response > 0.0
        latency[active] = 1.0 - response[active]
        return latency

    def transform_spikes(self, surfaces: np.ndarray) -> np.ndarray:
        return latency_to_spike_bins(
            self.transform_latency(surfaces),
            time_steps=int(self.config.time_steps),
        )


def event_surface_frontend_from_config(config: dict) -> EventSurfaceLatencyFrontend:
    return EventSurfaceLatencyFrontend(
        EventSurfaceLatencyFrontendConfig(
            input_channels=int(config.get("output_channels", 10)),
            time_steps=int(config.get("time_steps", 120)),
        )
    )
