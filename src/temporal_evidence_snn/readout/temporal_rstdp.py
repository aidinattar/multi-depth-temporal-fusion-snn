from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class SparseSpikingFeatureDataset:
    def __init__(self, dataset_file: str | Path, max_samples: int = 0) -> None:
        path = Path(dataset_file)
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature file: {path}")
        payload = np.load(path, allow_pickle=True).item()
        self.path = path
        labels = np.asarray(payload["labels"], dtype=np.int64)
        data = list(payload["data"])
        if int(max_samples) > 0:
            limit = min(int(max_samples), int(labels.shape[0]))
            labels = labels[:limit]
            data = data[:limit]
        self.labels = labels
        self.data = data
        shape = tuple(int(x) for x in payload["shape"])
        if len(shape) != 2:
            raise ValueError(f"Expected stored shape (N, D), got {shape}")
        self.shape = shape
        self.input_size = int(shape[1])
        self.max_time = float(payload.get("max_time", 1.0))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rec = self.data[int(idx)]
        if len(rec) == 0:
            indices = torch.empty((0,), dtype=torch.long)
            timestamps = torch.empty((0,), dtype=torch.float32)
        else:
            indices = torch.from_numpy(np.array(rec["indices"], dtype=np.int64, copy=True))
            timestamps = torch.from_numpy(np.array(rec["timestamps"], dtype=np.float32, copy=True))
        label = int(self.labels[int(idx)])
        return indices, timestamps, label


class DecisionMap:
    def __init__(self, n_neurons: int, n_nt_neurons: int, n_classes: int) -> None:
        self.n_neurons = int(n_neurons)
        self.n_nt_neurons = int(n_nt_neurons)
        self.n_classes = int(n_classes)
        self.n_neurons_per_class = int(self.n_neurons / self.n_classes)
        self.map_class = np.zeros((self.n_neurons,), dtype=np.int32)
        self.map_type = np.zeros((self.n_neurons,), dtype=np.int32)
        for i in range(self.n_neurons):
            self.map_class[i] = int(i / self.n_neurons_per_class)
            self.map_type[i] = 1 if i % self.n_neurons_per_class >= self.n_nt_neurons else 0

    def is_target_neuron(self, n: int) -> bool:
        return bool(self.map_type[int(n)] == 1)

    def get_target_neurons(self, y: int | None = None) -> np.ndarray:
        inds = []
        for i in range(self.n_neurons):
            if self.map_type[i] == 1 and (y is None or self.map_class[i] == int(y)):
                inds.append(i)
        return np.asarray(inds, dtype=np.int64)

    def get_class(self, n: int) -> int:
        return int(self.map_class[int(n)])


class EarlyStopper:
    def __init__(self, patience: int = 10) -> None:
        self.patience = int(patience)
        self.counter = 0
        self.max_acc = 0.0

    def early_stop(self, acc: float) -> bool:
        if float(acc) > float(self.max_acc):
            self.max_acc = float(acc)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def spike_sort(mem_pots: torch.Tensor, out_spks: torch.Tensor) -> torch.Tensor:
    order = sorted(
        range(int(out_spks.numel())),
        key=lambda i: (float(out_spks[i].item()), -float(mem_pots[i].item())),
    )
    return torch.as_tensor(order, device=out_spks.device, dtype=torch.long)


def stdp_additive(
    weights: torch.Tensor,
    in_spks: torch.Tensor,
    t_post: float,
    ap: float,
    am: float,
    error: float,
    max_time: float,
) -> torch.Tensor:
    causal = in_spks <= float(t_post)
    non_causal = in_spks < float(max_time)
    out = weights.clone()
    out[causal] += float(error) * float(ap)
    ltd_mask = (~causal) & non_causal
    out[ltd_mask] += float(error) * float(am)
    return out


class TemporalReadoutLayer:
    def __init__(self, layer_cfg: dict[str, Any], input_size: int, max_time: float, device: str):
        self.input_size = int(input_size)
        self.n_neurons = int(layer_cfg["n_neurons"])
        self.leak_tau = layer_cfg.get("leak_tau", None)
        self.w_min = float(layer_cfg.get("w_min", 0.0))
        self.w_max = float(layer_cfg.get("w_max", 1.0))
        self.max_time = float(max_time)
        self.train_mode = False
        self.device = torch.device(device)
        self.forward_mode = str(layer_cfg.get("forward_mode", "loop")).lower()

        if bool(layer_cfg.get("w_init_normal", True)):
            weight = torch.normal(
                mean=float(layer_cfg.get("w_init_mean", 0.5)),
                std=float(layer_cfg.get("w_init_std", 0.01)),
                size=(self.n_neurons, self.input_size),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            weight = torch.empty(
                (self.n_neurons, self.input_size),
                device=self.device,
                dtype=torch.float32,
            ).uniform_(self.w_min, self.w_max)
        self.weights = weight.clamp(self.w_min, self.w_max)
        self.w_norm = (
            self.weights.sum(dim=1).clone() if bool(layer_cfg.get("w_norm", False)) else None
        )
        self.thresholds = torch.full(
            (self.n_neurons,),
            float(layer_cfg["firing_threshold"]),
            device=self.device,
            dtype=torch.float32,
        )
        self.thresholds_train = self.thresholds.clone()
        self.clip_weights()
        self.normalize_weights()

    def clip_weights(self) -> None:
        self.weights.clamp_(self.w_min, self.w_max)

    def normalize_weights(self) -> None:
        if self.w_norm is None:
            return
        denom = self.weights.abs().sum(dim=1).clamp_min(1e-8)
        self.weights.mul_((self.w_norm / denom).unsqueeze(1))

    def train(self) -> None:
        self.train_mode = True

    def test(self) -> None:
        self.train_mode = False
        self.thresholds_train.copy_(self.thresholds)

    def forward_sparse_loop(
        self,
        indices: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        thresholds = self.thresholds_train if self.train_mode else self.thresholds
        mem_pots = torch.zeros((self.n_neurons,), device=self.device, dtype=torch.float32)
        in_spks = torch.full(
            (self.input_size,),
            float(self.max_time),
            device=self.device,
            dtype=torch.float32,
        )
        out_spks = torch.full(
            (self.n_neurons,),
            float(self.max_time),
            device=self.device,
            dtype=torch.float32,
        )
        active = torch.ones((self.n_neurons,), device=self.device, dtype=torch.bool)

        if int(timestamps.numel()) <= 0:
            return in_spks, out_spks, mem_pots

        idx_list = indices.tolist()
        ts_list = timestamps.tolist()
        curr_time = float(ts_list[0])
        n_events = len(ts_list)

        for i, (ind, time) in enumerate(zip(idx_list, ts_list)):
            time = float(time)
            if curr_time != time or i == (n_events - 1):
                if self.leak_tau is not None:
                    raise NotImplementedError("The temporal readout supports IF neurons only.")
                fire = active & (mem_pots > thresholds)
                out_spks[fire] = float(curr_time)
                active[fire] = False
                curr_time = float(time)
            if time == float(self.max_time):
                break
            in_spks[int(ind)] = float(time)
            if bool(active.any().item()):
                mem_pots[active] += self.weights[active, int(ind)]

        return in_spks, out_spks, mem_pots

    def forward_sparse_vectorized(
        self,
        indices: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.leak_tau is not None:
            raise NotImplementedError("The temporal readout supports IF neurons only.")

        thresholds = self.thresholds_train if self.train_mode else self.thresholds
        in_spks = torch.full(
            (self.input_size,),
            float(self.max_time),
            device=self.device,
            dtype=torch.float32,
        )
        out_spks = torch.full(
            (self.n_neurons,),
            float(self.max_time),
            device=self.device,
            dtype=torch.float32,
        )
        mem_pots = torch.zeros((self.n_neurons,), device=self.device, dtype=torch.float32)

        if int(timestamps.numel()) <= 0:
            return in_spks, out_spks, mem_pots

        indices = indices.to(device=self.device, dtype=torch.long)
        timestamps = timestamps.to(device=self.device, dtype=torch.float32)
        finite = timestamps < float(self.max_time)
        if not bool(finite.any().item()):
            return in_spks, out_spks, mem_pots
        indices = indices[finite]
        timestamps = timestamps[finite]
        if int(indices.numel()) <= 0:
            return in_spks, out_spks, mem_pots

        # Feature files are timestamp-sorted. Keep the fallback robust for any
        # future exporter that might not preserve this invariant.
        if bool((timestamps[1:] < timestamps[:-1]).any().item()):
            order = torch.argsort(timestamps)
            timestamps = timestamps[order]
            indices = indices[order]

        in_spks[indices] = timestamps
        weights_events = self.weights[:, indices]
        cumulative = torch.cumsum(weights_events, dim=1)
        group_times, counts = torch.unique_consecutive(timestamps, return_counts=True)
        group_ends = torch.cumsum(counts, dim=0) - 1
        group_mem = cumulative[:, group_ends]

        # Membrane potential used for tie-breaking is the potential at spike
        # time for firing neurons, otherwise the final potential.
        mem_pots.copy_(group_mem[:, -1])
        crossed = group_mem > thresholds[:, None]
        fired = crossed.any(dim=1)
        if bool(fired.any().item()):
            first_group = crossed.to(torch.int64).argmax(dim=1)
            rows = torch.arange(self.n_neurons, device=self.device, dtype=torch.long)
            fired_rows = rows[fired]
            fired_groups = first_group[fired]
            out_spks[fired_rows] = group_times[fired_groups]
            mem_pots[fired_rows] = group_mem[fired_rows, fired_groups]
        return in_spks, out_spks, mem_pots

    def forward_sparse(
        self,
        indices: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.forward_mode in {"vectorized", "fast"}:
            return self.forward_sparse_vectorized(indices, timestamps)
        if self.forward_mode in {"loop", "reference"}:
            return self.forward_sparse_loop(indices, timestamps)
        raise ValueError(f"Unsupported reference head forward_mode={self.forward_mode!r}")


@dataclass
class AdditiveRule:
    max_time: float = float("inf")

    def apply(
        self,
        weights: torch.Tensor,
        in_spks: torch.Tensor,
        t_post: float,
        ap: float,
        am: float,
        error: float,
    ) -> torch.Tensor:
        return stdp_additive(weights, in_spks, t_post, ap, am, error, self.max_time)


class TemporalPrototypeRSTDP:
    """Temporal-target R-STDP with sparse hard-negative prototypes."""

    def __init__(
        self,
        layer: TemporalReadoutLayer,
        stdp_rule: AdditiveRule,
        *,
        n_classes: int,
        ap: float,
        am: float,
        anti_ap: float | None = None,
        anti_am: float | None = None,
        adaptive_lr: bool = True,
        annealing: float = 1.0,
        max_time: float = 1.0,
        correct_margin: float = 0.0,
        non_target_scale: float = 1.0,
        update_scale: float = 1.0,
        temporal_margin: float = 0.005,
        temporal_scale_cap: float = 0.02,
        temporal_contrast_target_scale: float = 1.0,
        temporal_hard_negative_k: int = 3,
        temporal_target_prototypes: int = 1,
        temporal_negative_prototypes: int = 1,
        temporal_prototype_decay: float = 0.5,
        temporal_floor_scale: float = 0.0,
        temporal_stability_min_spikes: float = 45.0,
        temporal_guard_anti_scale: float = 0.25,
        temporal_guard_target_scale: float = 1.25,
        temporal_correct_update_scale: float = 1.0,
        temporal_confident_correct_scale: float | None = None,
        temporal_anti_update_scale: float = 1.0,
    ) -> None:
        self.layer = layer
        self.stdp_rule = stdp_rule
        self.n_classes = int(n_classes)
        self.max_time = float(max_time)
        self.ap_init = float(ap)
        self.am_init = float(am)
        self.anti_ap_init = -float(ap) if anti_ap is None else float(anti_ap)
        self.anti_am_init = -float(am) if anti_am is None else float(anti_am)
        self.adaptive_lr = bool(adaptive_lr)
        self.annealing = float(annealing)
        self.correct_margin = float(correct_margin)
        self.non_target_scale = float(non_target_scale)
        self.update_scale = float(update_scale)
        self.temporal_margin = float(temporal_margin)
        self.temporal_scale_cap = float(temporal_scale_cap)
        self.temporal_contrast_target_scale = float(temporal_contrast_target_scale)
        self.temporal_hard_negative_k = int(temporal_hard_negative_k)
        self.temporal_target_prototypes = max(1, int(temporal_target_prototypes))
        self.temporal_negative_prototypes = max(1, int(temporal_negative_prototypes))
        self.temporal_prototype_decay = max(0.0, float(temporal_prototype_decay))
        self.temporal_floor_scale = float(temporal_floor_scale)
        self.temporal_stability_min_spikes = float(temporal_stability_min_spikes)
        self.temporal_guard_anti_scale = float(temporal_guard_anti_scale)
        self.temporal_guard_target_scale = float(temporal_guard_target_scale)
        self.temporal_correct_update_scale = float(temporal_correct_update_scale)
        self.temporal_confident_correct_scale = (
            float(temporal_correct_update_scale)
            if temporal_confident_correct_scale is None
            else float(temporal_confident_correct_scale)
        )
        self.temporal_anti_update_scale = float(temporal_anti_update_scale)
        self.accuracy_trace: list[int] = []
        self.ap = self.ap_init
        self.am = self.am_init
        self.anti_ap = self.anti_ap_init
        self.anti_am = self.anti_am_init
        self._set_adaptive_rates(mean_acc=1.0 / max(1, self.n_classes))

    def _set_adaptive_rates(self, *, mean_acc: float) -> None:
        lr_mod = (1.0 - float(mean_acc)) if self.adaptive_lr else 1.0
        anti_lr_mod = float(mean_acc) if self.adaptive_lr else 1.0
        self.ap = self.ap_init * lr_mod
        self.am = self.am_init * lr_mod
        self.anti_ap = self.anti_ap_init * anti_lr_mod
        self.anti_am = self.anti_am_init * anti_lr_mod

    def anneal(self) -> None:
        self.ap_init *= self.annealing
        self.am_init *= self.annealing
        self.anti_ap_init *= self.annealing
        self.anti_am_init *= self.annealing
        mean_acc = (
            float(np.mean(self.accuracy_trace))
            if self.accuracy_trace
            else 1.0 / max(1, self.n_classes)
        )
        self._set_adaptive_rates(mean_acc=mean_acc)
        self.accuracy_trace = []

    def _margin(self, out_spks: torch.Tensor, mem_pots: torch.Tensor) -> float:
        order = spike_sort(mem_pots, out_spks)
        if int(order.numel()) < 2:
            return 0.0
        first = int(order[0].item())
        second = int(order[1].item())
        dt = float(out_spks[second].item()) - float(out_spks[first].item())
        if abs(dt) > 1e-12:
            return float(dt)
        return float(mem_pots[first].item()) - float(mem_pots[second].item())

    def _class_winner(
        self,
        out_spks: torch.Tensor,
        mem_pots: torch.Tensor,
        decision_map: DecisionMap,
        class_ind: int,
    ) -> int:
        start = int(class_ind) * decision_map.n_neurons_per_class
        end = start + decision_map.n_neurons_per_class
        inds = torch.arange(start, end, device=out_spks.device, dtype=torch.long)
        sorted_local = spike_sort(mem_pots[inds], out_spks[inds])
        return int(inds[sorted_local[0]].item())

    def _class_prototypes(
        self,
        out_spks: torch.Tensor,
        mem_pots: torch.Tensor,
        decision_map: DecisionMap,
        class_ind: int,
        count: int,
    ) -> list[int]:
        start = int(class_ind) * decision_map.n_neurons_per_class
        end = start + decision_map.n_neurons_per_class
        inds = torch.arange(start, end, device=out_spks.device, dtype=torch.long)
        sorted_local = spike_sort(mem_pots[inds], out_spks[inds])
        return [
            int(inds[int(local_idx.item())].item())
            for local_idx in sorted_local[: max(1, int(count))]
        ]

    def _temporal_target_hard_negative_multiproto_updates(
        self,
        *,
        out_spks: torch.Tensor,
        mem_pots: torch.Tensor,
        decision_map: DecisionMap,
        target_ind: int,
    ) -> tuple[list[tuple[int, float, float, float]], float, float]:
        """Build sparse updates for target and hard-negative prototypes."""
        class_winners: list[int] = []
        for cls in range(decision_map.n_classes):
            class_winners.append(self._class_winner(out_spks, mem_pots, decision_map, int(cls)))
        if not class_winners:
            return [], 0.0, 0.0

        winners_t = torch.as_tensor(class_winners, device=out_spks.device, dtype=torch.long)
        winner_times = out_spks[winners_t]
        finite = winner_times[winner_times < float(self.max_time)]
        if int(finite.numel()) == 0:
            base_time = float(self.max_time) - 0.5 * float(self.temporal_margin)
        else:
            base_time = min(
                float(finite.mean().item()),
                float(self.max_time) - 0.5 * float(self.temporal_margin),
            )

        half_margin = 0.5 * float(self.temporal_margin)
        target_desired = max(0.0, base_time - half_margin)
        non_target_desired = min(float(self.max_time), base_time + half_margin)
        cap = max(float(self.temporal_scale_cap), 0.0)
        floor = max(float(self.temporal_floor_scale), 0.0)
        fired_count = float((out_spks < float(self.max_time)).sum().item())
        unstable = fired_count < float(self.temporal_stability_min_spikes)
        anti_guard = max(0.0, float(self.temporal_guard_anti_scale)) if unstable else 1.0
        target_guard = max(0.0, float(self.temporal_guard_target_scale)) if unstable else 1.0

        updates: list[tuple[int, float, float, float]] = []
        scales: list[float] = []
        decay = float(self.temporal_prototype_decay)

        target_prototypes = self._class_prototypes(
            out_spks,
            mem_pots,
            decision_map,
            int(target_ind),
            int(self.temporal_target_prototypes),
        )
        if not target_prototypes:
            return [], 0.0, 0.0

        for rank, target_neuron in enumerate(target_prototypes):
            target_time = float(out_spks[int(target_neuron)].item())
            target_error = max(0.0, target_time - target_desired)
            proto_scale = decay**rank
            scale = min(
                cap,
                max(floor, target_error / max(float(self.max_time), 1e-8))
                * float(self.temporal_contrast_target_scale)
                * target_guard
                * proto_scale,
            )
            if scale <= 0.0:
                continue
            updates.append((int(target_neuron), self.ap, self.am, float(scale)))
            scales.append(float(scale))

        candidates: list[tuple[float, float, int]] = []
        for cls, class_winner in enumerate(class_winners):
            if int(cls) == int(target_ind):
                continue
            class_time = float(out_spks[int(class_winner)].item())
            non_target_error = max(0.0, non_target_desired - class_time)
            candidates.append((float(non_target_error), -float(class_time), int(cls)))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        class_k = max(0, min(int(self.temporal_hard_negative_k), len(candidates)))
        for _class_error, _neg_time, cls in candidates[:class_k]:
            class_prototypes = self._class_prototypes(
                out_spks,
                mem_pots,
                decision_map,
                int(cls),
                int(self.temporal_negative_prototypes),
            )
            for rank, class_neuron in enumerate(class_prototypes):
                class_time = float(out_spks[int(class_neuron)].item())
                non_target_error = max(0.0, non_target_desired - class_time)
                proto_scale = decay**rank
                scale = min(
                    cap,
                    max(floor, float(non_target_error) / max(float(self.max_time), 1e-8))
                    * float(self.non_target_scale)
                    * anti_guard
                    * proto_scale,
                )
                if scale <= 0.0:
                    continue
                updates.append((int(class_neuron), self.anti_ap, self.anti_am, float(scale)))
                scales.append(float(scale))

        return (
            updates,
            float(max(scales) if scales else 0.0),
            float(np.mean(scales) if scales else 0.0),
        )

    def _apply_update(
        self,
        *,
        neuron: int,
        in_spks: torch.Tensor,
        out_spks: torch.Tensor,
        ap: float,
        am: float,
        scale: float = 1.0,
    ) -> bool:
        n = int(neuron)
        effective_scale = float(scale)
        if float(ap) < 0.0 and float(am) > 0.0:
            effective_scale *= float(self.temporal_anti_update_scale)
        updated = self.stdp_rule.apply(
            self.layer.weights[n],
            in_spks,
            float(out_spks[n].item()),
            float(ap) * effective_scale * self.update_scale,
            float(am) * effective_scale * self.update_scale,
            1.0,
        )
        self.layer.weights[n].copy_(updated)
        return True

    def _scale_temporal_updates_for_correct_sample(
        self,
        updates: list[tuple[int, float, float, float]],
        *,
        correct_class: bool,
        correct_target: bool,
        margin: float,
    ) -> list[tuple[int, float, float, float]]:
        """Reduce dense temporal updates once a sample is already solved.

        Dense temporal-target rules keep moving every class winner on every
        sample. That is useful early, but after a sample is classified correctly
        it can keep perturbing weights until firing statistics drift. This gate
        preserves learning on mistakes while making correct samples less
        plastic. If correct_margin is set, confidently correct target-neuron
        wins can use an even smaller scale.
        """
        if not correct_class:
            return updates

        scale = float(self.temporal_correct_update_scale)
        if (
            correct_target
            and float(self.correct_margin) > 0.0
            and float(margin) >= float(self.correct_margin)
        ):
            scale = float(self.temporal_confident_correct_scale)
        if scale <= 0.0:
            return []
        if abs(scale - 1.0) <= 1e-12:
            return updates
        return [
            (int(n_ind), float(ap), float(am), float(update_scale) * scale)
            for n_ind, ap, am, update_scale in updates
        ]

    def step(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target_ind: int,
        decision_map: DecisionMap,
    ) -> dict[str, float]:
        in_spks, out_spks, mem_pots = outputs
        winner = int(spike_sort(mem_pots, out_spks)[0].item())
        predicted = decision_map.get_class(winner)
        correct_class = int(predicted) == int(target_ind)
        correct_target = correct_class and decision_map.is_target_neuron(winner)
        self.accuracy_trace.append(1 if correct_class else 0)

        margin = self._margin(out_spks, mem_pots)
        updates, temporal_violation_max, temporal_violation_mean = (
            self._temporal_target_hard_negative_multiproto_updates(
                out_spks=out_spks,
                mem_pots=mem_pots,
                decision_map=decision_map,
                target_ind=int(target_ind),
            )
        )
        updates = self._scale_temporal_updates_for_correct_sample(
            updates,
            correct_class=bool(correct_class),
            correct_target=bool(correct_target),
            margin=float(margin),
        )

        applied: set[int] = set()
        for neuron, ap, am, scale in updates:
            self._apply_update(
                neuron=int(neuron),
                in_spks=in_spks,
                out_spks=out_spks,
                ap=float(ap),
                am=float(am),
                scale=float(scale),
            )
            applied.add(int(neuron))
        if applied:
            self.layer.clip_weights()
            self.layer.normalize_weights()

        return {
            "updated_neuron_ratio": float(len(applied) / max(1, int(out_spks.numel()))),
            "A_plus": float(self.ap),
            "A_minus": float(self.am),
            "anti_A_plus": float(self.anti_ap),
            "anti_A_minus": float(self.anti_am),
            "rstdp_margin": float(margin),
            "rstdp_correct_class": float(correct_class),
            "rstdp_correct_target": float(correct_target),
            "rstdp_temporal_violation_max": float(temporal_violation_max),
            "rstdp_temporal_violation_mean": float(temporal_violation_mean),
        }


class CompetitionRegularizerTwoTorch:
    def __init__(self, layer: TemporalReadoutLayer, thr_lr: float, thr_anneal: float = 1.0) -> None:
        self.layer = layer
        self.thr_lr = float(thr_lr)
        self.thr_anneal = float(thr_anneal)
        self.thresholds = self.layer.thresholds.clone()
        self.thr_min = float(self.layer.thresholds.min().item())
        self.thr_max = None

    def on_epoch_start(self) -> None:
        self.thresholds.copy_(self.layer.thresholds)

    def on_epoch_end(self) -> None:
        self.thr_lr *= self.thr_anneal

    def compute(self, y: int, decision_map: DecisionMap) -> None:
        n_inds_t = decision_map.get_target_neurons(int(y))
        self.layer.thresholds_train.copy_(self.layer.thresholds)
        if len(n_inds_t) > 0:
            inds_t = torch.as_tensor(n_inds_t, device=self.layer.device, dtype=torch.long)
            self.layer.thresholds_train[inds_t] = self.thresholds[inds_t]

    def step(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        y: int,
        decision_map: DecisionMap,
    ) -> dict[str, float]:
        if self.thr_lr <= 0:
            return {}
        _, out_spks, mem_pots = outputs
        n_inds_np = decision_map.get_target_neurons(int(y))
        if len(n_inds_np) <= 1:
            return {}
        n_inds = torch.as_tensor(n_inds_np, device=out_spks.device, dtype=torch.long)
        sorted_local = spike_sort(mem_pots[n_inds], out_spks[n_inds])
        winner = int(n_inds[sorted_local[0]].item())
        losers = n_inds[sorted_local[1:]]
        if decision_map.is_target_neuron(winner):
            self.thresholds[winner] += self.thr_lr * (len(n_inds_np) - 1) / len(n_inds_np)
            if int(losers.numel()) > 0:
                self.thresholds[losers] -= self.thr_lr * 1.0 / len(n_inds_np)
            self.thresholds.clamp_(min=self.thr_min)
        return {
            "thr_lr": float(self.thr_lr),
            "threshold_mean": float(self.thresholds.mean().item()),
            "threshold_std": float(self.thresholds.std(unbiased=False).item()),
        }


def _epoch_path(output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / name


class TemporalReadout:
    def __init__(
        self,
        *,
        n_classes: int,
        layer: TemporalReadoutLayer,
        optimizer: TemporalPrototypeRSTDP,
        regularizer: CompetitionRegularizerTwoTorch,
        config: dict[str, Any],
        output_dir: str | Path | None = None,
    ) -> None:
        self.layer = layer
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.config = dict(config)
        self.decision_map = DecisionMap(
            n_neurons=self.layer.n_neurons,
            n_nt_neurons=int(self.config.get("nt_neurons", 0)),
            n_classes=int(n_classes),
        )
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.metrics_path = (
            None
            if self.output_dir is None
            else _epoch_path(self.output_dir, "readout_metrics.jsonl")
        )
        self.summary_path = (
            None
            if self.output_dir is None
            else _epoch_path(self.output_dir, "readout_summary.json")
        )
        self.log_path = (
            None if self.output_dir is None else _epoch_path(self.output_dir, "readout.log")
        )
        self.best_path = (
            None
            if self.output_dir is None
            else _epoch_path(self.output_dir / "checkpoints", "best_readout.pt")
        )
        self.progress_every = int(self.config.get("progress_every", 1000))
        self.consolidation_start_epoch = int(self.config.get("consolidation_start_epoch", -1))
        self.consolidation_update_scale = float(self.config.get("consolidation_update_scale", 1.0))
        self.consolidation_correct_update_scale = float(
            self.config.get("consolidation_correct_update_scale", 1.0)
        )
        self.consolidation_confident_correct_scale = float(
            self.config.get("consolidation_confident_correct_scale", 1.0)
        )
        self.consolidation_anti_update_scale = float(
            self.config.get("consolidation_anti_update_scale", 1.0)
        )
        self._base_optimizer_update_scale = float(getattr(self.optimizer, "update_scale", 1.0))
        self._base_temporal_correct_update_scale = float(
            getattr(self.optimizer, "temporal_correct_update_scale", 1.0)
        )
        self._base_temporal_confident_correct_scale = float(
            getattr(self.optimizer, "temporal_confident_correct_scale", 1.0)
        )
        self._base_temporal_anti_update_scale = float(
            getattr(self.optimizer, "temporal_anti_update_scale", 1.0)
        )

    def _log(self, msg: str) -> None:
        print(msg, flush=True)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def _append_epoch_metrics(self, payload: dict[str, Any]) -> None:
        if self.metrics_path is None:
            return
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _write_summary(self, payload: dict[str, Any]) -> None:
        if self.summary_path is None:
            return
        self.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_best(self, *, epoch: int, monitor_acc: float, monitor_name: str) -> None:
        if self.best_path is None:
            return
        torch.save(
            {
                "epoch": int(epoch),
                "monitor_acc": float(monitor_acc),
                "monitor_name": str(monitor_name),
                "weights": self.layer.weights.detach().cpu(),
                "thresholds": self.layer.thresholds.detach().cpu(),
                "thresholds_train": self.layer.thresholds_train.detach().cpu(),
                "regularizer_thresholds": (
                    self.regularizer.thresholds.detach().cpu()
                    if hasattr(self.regularizer, "thresholds")
                    else None
                ),
            },
            self.best_path,
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
        """Load a readout checkpoint written by the training pipeline."""

        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=self.layer.device,
            weights_only=False,
        )
        required = {"weights", "thresholds", "thresholds_train"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError("Readout checkpoint is missing: " + ", ".join(sorted(missing)))

        weights = checkpoint["weights"].to(
            device=self.layer.device,
            dtype=self.layer.weights.dtype,
        )
        thresholds = checkpoint["thresholds"].to(
            device=self.layer.device,
            dtype=self.layer.thresholds.dtype,
        )
        thresholds_train = checkpoint["thresholds_train"].to(
            device=self.layer.device,
            dtype=self.layer.thresholds_train.dtype,
        )
        if weights.shape != self.layer.weights.shape:
            raise ValueError(
                f"Checkpoint weights have shape {tuple(weights.shape)}; "
                f"expected {tuple(self.layer.weights.shape)}."
            )
        if thresholds.shape != self.layer.thresholds.shape:
            raise ValueError(
                f"Checkpoint thresholds have shape {tuple(thresholds.shape)}; "
                f"expected {tuple(self.layer.thresholds.shape)}."
            )
        if thresholds_train.shape != self.layer.thresholds_train.shape:
            raise ValueError(
                "Checkpoint training thresholds have shape "
                f"{tuple(thresholds_train.shape)}; expected "
                f"{tuple(self.layer.thresholds_train.shape)}."
            )

        self.layer.weights.copy_(weights)
        self.layer.thresholds.copy_(thresholds)
        self.layer.thresholds_train.copy_(thresholds_train)
        regularizer_thresholds = checkpoint.get("regularizer_thresholds")
        if regularizer_thresholds is not None:
            regularizer_thresholds = regularizer_thresholds.to(
                device=self.layer.device,
                dtype=self.regularizer.thresholds.dtype,
            )
            if regularizer_thresholds.shape != self.regularizer.thresholds.shape:
                raise ValueError(
                    "Checkpoint regularizer thresholds have shape "
                    f"{tuple(regularizer_thresholds.shape)}; expected "
                    f"{tuple(self.regularizer.thresholds.shape)}."
                )
            self.regularizer.thresholds.copy_(regularizer_thresholds)
        return checkpoint

    def _apply_phase_schedule(self, epoch: int) -> str:
        if self.consolidation_start_epoch < 0 or int(epoch) < self.consolidation_start_epoch:
            phase = "plastic"
            update_scale = self._base_optimizer_update_scale
            correct_scale = self._base_temporal_correct_update_scale
            confident_scale = self._base_temporal_confident_correct_scale
            anti_update_scale = self._base_temporal_anti_update_scale
        else:
            phase = "consolidation"
            update_scale = self._base_optimizer_update_scale * self.consolidation_update_scale
            correct_scale = (
                self._base_temporal_correct_update_scale * self.consolidation_correct_update_scale
            )
            confident_scale = (
                self._base_temporal_confident_correct_scale
                * self.consolidation_confident_correct_scale
            )
            anti_update_scale = (
                self._base_temporal_anti_update_scale * self.consolidation_anti_update_scale
            )
        if hasattr(self.optimizer, "update_scale"):
            self.optimizer.update_scale = float(update_scale)
        if hasattr(self.optimizer, "temporal_correct_update_scale"):
            self.optimizer.temporal_correct_update_scale = float(correct_scale)
        if hasattr(self.optimizer, "temporal_confident_correct_scale"):
            self.optimizer.temporal_confident_correct_scale = float(confident_scale)
        if hasattr(self.optimizer, "temporal_anti_update_scale"):
            self.optimizer.temporal_anti_update_scale = float(anti_update_scale)
        return phase

    def _predict_sample(
        self, indices: torch.Tensor, timestamps: torch.Tensor
    ) -> tuple[int, float, float]:
        outputs = self.layer.forward_sparse(
            indices.to(self.layer.device), timestamps.to(self.layer.device)
        )
        _, out_spks, mem_pots = outputs
        winner = int(spike_sort(mem_pots, out_spks)[0].item())
        pred = self.decision_map.get_class(winner)
        winner_time = float(out_spks[winner].item())
        return pred, winner_time, float(winner_time >= self.layer.max_time)

    def predict(self, dataset: SparseSpikingFeatureDataset, *, name: str = "dataset") -> float:
        self.layer.test()
        correct = 0
        self._log(f"[predict:{name}] start samples={len(dataset)}")
        for idx in range(len(dataset)):
            indices, timestamps, y = dataset.sample(idx)
            pred, _, _ = self._predict_sample(indices, timestamps)
            correct += int(pred == int(y))
            if self.progress_every > 0 and (idx + 1) % self.progress_every == 0:
                self._log(
                    f"[predict:{name}] {idx + 1}/{len(dataset)} "
                    f"acc={float(correct) / float(idx + 1):.4f}"
                )
        self._log(f"[predict:{name}] done acc={float(correct) / max(1, len(dataset)):.4f}")
        return float(correct) / max(1, len(dataset))

    def fit(
        self,
        train_dataset: SparseSpikingFeatureDataset,
        val_dataset: SparseSpikingFeatureDataset | None = None,
        test_dataset: SparseSpikingFeatureDataset | None = None,
    ) -> dict[str, Any]:
        epochs = int(self.config["epochs"])
        early_stopper = None
        if int(self.config.get("early_stopping", 0)) > 0:
            early_stopper = EarlyStopper(int(self.config["early_stopping"]))

        last_summary: dict[str, Any] = {
            "status": "completed",
            "epochs_requested": int(epochs),
            "epochs_completed": 0,
            "best_val_acc": None,
            "best_test_acc": None,
            "best_monitor_acc": None,
            "best_monitor_epoch": None,
            "best_monitor_name": None,
            "final_train_acc": None,
            "final_val_acc": None,
            "final_test_acc": None,
        }
        best_monitor_acc = -float("inf")
        best_monitor_epoch = None
        best_monitor_name = None

        self._log(
            f"[fit] start epochs={epochs} train_samples={len(train_dataset)} "
            f"test_samples={0 if test_dataset is None else len(test_dataset)}"
        )

        for epoch in range(epochs):
            self._log(f"[epoch:{epoch}] train start")
            phase = self._apply_phase_schedule(epoch)
            self.layer.train()
            self.regularizer.on_epoch_start()

            train_acc = 0
            spike_counts: list[float] = []
            winner_times: list[float] = []
            winner_silent: list[float] = []
            updated_stats: list[dict[str, float]] = []

            for sample_idx in range(len(train_dataset)):
                indices, timestamps, y = train_dataset.sample(sample_idx)
                indices = indices.to(self.layer.device)
                timestamps = timestamps.to(self.layer.device)

                self.regularizer.compute(y, self.decision_map)
                outputs = self.layer.forward_sparse(indices, timestamps)
                _, out_spks, mem_pots = outputs
                winner = int(spike_sort(mem_pots, out_spks)[0].item())
                predicted = self.decision_map.get_class(winner)
                train_acc += int(predicted == int(y))
                spike_counts.append(float((out_spks < self.layer.max_time).sum().item()))
                winner_times.append(float(out_spks[winner].item()))
                winner_silent.append(float(out_spks[winner].item() >= self.layer.max_time))

                reg_stats = self.regularizer.step(outputs, y, self.decision_map)
                opt_stats = self.optimizer.step(outputs, y, self.decision_map)
                merged = {}
                merged.update(reg_stats)
                merged.update(opt_stats)
                updated_stats.append(merged)
                if self.progress_every > 0 and (sample_idx + 1) % self.progress_every == 0:
                    self._log(
                        f"[epoch:{epoch}] train {sample_idx + 1}/{len(train_dataset)} "
                        f"acc={float(train_acc) / float(sample_idx + 1):.4f}"
                    )

            train_acc = float(train_acc) / max(1, len(train_dataset))
            self.optimizer.anneal()
            self.regularizer.on_epoch_end()

            val_acc = None
            test_acc = None
            stop_kind = None

            if val_dataset is not None:
                val_acc = self.predict(val_dataset, name=f"val:e{epoch}")
                if early_stopper is not None and early_stopper.early_stop(val_acc):
                    stop_kind = "early_stop"
            if test_dataset is not None:
                test_acc = self.predict(test_dataset, name=f"test:e{epoch}")

            monitor_name = None
            monitor_acc = None
            if val_acc is not None:
                monitor_name = "val_acc"
                monitor_acc = float(val_acc)
            elif test_acc is not None:
                monitor_name = "test_acc"
                monitor_acc = float(test_acc)
            if monitor_acc is not None and monitor_acc > best_monitor_acc:
                best_monitor_acc = float(monitor_acc)
                best_monitor_epoch = int(epoch)
                best_monitor_name = str(monitor_name)
                self._save_best(
                    epoch=int(epoch),
                    monitor_acc=float(monitor_acc),
                    monitor_name=str(monitor_name),
                )

            mean_update_ratio = 0.0
            mean_temporal_violation = 0.0
            max_temporal_violation = 0.0
            mean_rstdp_margin = 0.0
            mean_rstdp_correct_class = 0.0
            mean_rstdp_correct_target = 0.0
            if updated_stats:
                vals = [
                    float(s.get("updated_neuron_ratio", 0.0))
                    for s in updated_stats
                    if "updated_neuron_ratio" in s
                ]
                if vals:
                    mean_update_ratio = float(np.mean(vals))
                temporal_vals = [
                    float(s.get("rstdp_temporal_violation_mean", 0.0))
                    for s in updated_stats
                    if "rstdp_temporal_violation_mean" in s
                ]
                if temporal_vals:
                    mean_temporal_violation = float(np.mean(temporal_vals))
                temporal_max_vals = [
                    float(s.get("rstdp_temporal_violation_max", 0.0))
                    for s in updated_stats
                    if "rstdp_temporal_violation_max" in s
                ]
                if temporal_max_vals:
                    max_temporal_violation = float(np.max(temporal_max_vals))
                margin_vals = [
                    float(s.get("rstdp_margin", 0.0)) for s in updated_stats if "rstdp_margin" in s
                ]
                if margin_vals:
                    mean_rstdp_margin = float(np.mean(margin_vals))
                correct_class_vals = [
                    float(s.get("rstdp_correct_class", 0.0))
                    for s in updated_stats
                    if "rstdp_correct_class" in s
                ]
                if correct_class_vals:
                    mean_rstdp_correct_class = float(np.mean(correct_class_vals))
                correct_target_vals = [
                    float(s.get("rstdp_correct_target", 0.0))
                    for s in updated_stats
                    if "rstdp_correct_target" in s
                ]
                if correct_target_vals:
                    mean_rstdp_correct_target = float(np.mean(correct_target_vals))

            epoch_payload = {
                "epoch": int(epoch),
                "train_acc": float(train_acc),
                "val_acc": None if val_acc is None else float(val_acc),
                "test_acc": None if test_acc is None else float(test_acc),
                "stop_kind": stop_kind,
                "optimizer": {
                    "ap": float(self.optimizer.ap),
                    "am": float(self.optimizer.am),
                    "anti_ap": float(self.optimizer.anti_ap),
                    "anti_am": float(self.optimizer.anti_am),
                    "annealing": float(self.optimizer.annealing),
                    "phase": str(phase),
                    "update_scale": float(getattr(self.optimizer, "update_scale", 1.0)),
                    "temporal_correct_update_scale": float(
                        getattr(self.optimizer, "temporal_correct_update_scale", 1.0)
                    ),
                    "temporal_confident_correct_scale": float(
                        getattr(
                            self.optimizer,
                            "temporal_confident_correct_scale",
                            1.0,
                        )
                    ),
                    "temporal_anti_update_scale": float(
                        getattr(self.optimizer, "temporal_anti_update_scale", 1.0)
                    ),
                },
                "regularizer": {
                    "threshold_mean": float(
                        getattr(self.regularizer, "thresholds", self.layer.thresholds).mean().item()
                    )
                    if hasattr(
                        getattr(self.regularizer, "thresholds", self.layer.thresholds), "mean"
                    )
                    else None,
                    "threshold_std": float(
                        getattr(self.regularizer, "thresholds", self.layer.thresholds)
                        .std(unbiased=False)
                        .item()
                    )
                    if hasattr(
                        getattr(self.regularizer, "thresholds", self.layer.thresholds), "std"
                    )
                    else None,
                    "thr_lr": float(getattr(self.regularizer, "thr_lr", 0.0)),
                },
                "layers": [
                    {
                        "layer_index": 0,
                        "n_neurons": int(self.layer.n_neurons),
                        "weight_mean": float(self.layer.weights.mean().item()),
                        "weight_std": float(self.layer.weights.std(unbiased=False).item()),
                        "weight_min": float(self.layer.weights.min().item()),
                        "weight_max": float(self.layer.weights.max().item()),
                        "spike_count_mean": float(np.mean(spike_counts)) if spike_counts else 0.0,
                        "spike_count_std": float(np.std(spike_counts)) if spike_counts else 0.0,
                    }
                ],
                "readout": {
                    "output_spike_count_mean": float(np.mean(spike_counts))
                    if spike_counts
                    else 0.0,
                    "output_spike_count_std": float(np.std(spike_counts)) if spike_counts else 0.0,
                    "winner_time_mean": float(np.mean(winner_times)) if winner_times else 0.0,
                    "winner_time_std": float(np.std(winner_times)) if winner_times else 0.0,
                    "winner_silent_ratio": float(np.mean(winner_silent)) if winner_silent else 0.0,
                    "updated_neuron_ratio_mean": float(mean_update_ratio),
                    "rstdp_margin_mean": float(mean_rstdp_margin),
                    "rstdp_correct_class_mean": float(mean_rstdp_correct_class),
                    "rstdp_correct_target_mean": float(mean_rstdp_correct_target),
                    "rstdp_temporal_violation_mean": float(mean_temporal_violation),
                    "rstdp_temporal_violation_max": float(max_temporal_violation),
                },
            }
            self._append_epoch_metrics(epoch_payload)
            self._log(
                f"epoch={epoch} train_acc={train_acc:.4f} "
                f"test_acc={'n/a' if test_acc is None else f'{test_acc:.4f}'} "
                f"spk_mean={epoch_payload['readout']['output_spike_count_mean']:.2f} "
                f"winner_silent={epoch_payload['readout']['winner_silent_ratio']:.4f}"
            )

            last_summary = {
                "status": stop_kind if stop_kind is not None else "completed",
                "epochs_requested": int(epochs),
                "epochs_completed": int(epoch + 1),
                "best_val_acc": None if early_stopper is None else float(early_stopper.max_acc),
                "best_test_acc": None
                if test_acc is None
                else float(
                    max(
                        [
                            float(test_acc),
                            float(last_summary.get("best_test_acc") or -float("inf")),
                        ]
                    )
                ),
                "best_monitor_acc": None if best_monitor_epoch is None else float(best_monitor_acc),
                "best_monitor_epoch": best_monitor_epoch,
                "best_monitor_name": best_monitor_name,
                "final_train_acc": float(train_acc),
                "final_val_acc": None if val_acc is None else float(val_acc),
                "final_test_acc": None if test_acc is None else float(test_acc),
            }

            if stop_kind is not None:
                break

        self._write_summary(last_summary)
        return last_summary


def build_readout_from_config(
    config: dict[str, Any],
    *,
    input_size: int,
    n_classes: int,
    max_time: float,
    device: str,
    output_dir: str | Path | None,
) -> TemporalReadout:
    if len(config["network"]) != 1:
        raise ValueError("The temporal readout requires exactly one fully connected layer.")

    optimizer_config = config["optimizer"]
    method = str(optimizer_config.get("method", "")).lower()
    mode = str(optimizer_config.get("mode", "")).lower()
    stdp_name = str(optimizer_config.get("stdp", "")).lower()
    if method != "rstdp":
        raise ValueError("optimizer.method must be 'rstdp'.")
    if mode != "temporal_target_hard_negative_multiproto":
        raise ValueError("optimizer.mode must be 'temporal_target_hard_negative_multiproto'.")
    if stdp_name != "additive":
        raise ValueError("optimizer.stdp must be 'additive'.")

    layer = TemporalReadoutLayer(
        config["network"][0],
        input_size=input_size,
        max_time=max_time,
        device=device,
    )
    ignore_silent = bool(optimizer_config.get("ignore_silent", False))
    stdp_rule = AdditiveRule(max_time=max_time if ignore_silent else float("inf"))
    optimizer = TemporalPrototypeRSTDP(
        layer=layer,
        stdp_rule=stdp_rule,
        n_classes=int(n_classes),
        ap=float(optimizer_config["ap"]),
        am=float(optimizer_config["am"]),
        anti_ap=(None if "anti_ap" not in optimizer_config else float(optimizer_config["anti_ap"])),
        anti_am=(None if "anti_am" not in optimizer_config else float(optimizer_config["anti_am"])),
        adaptive_lr=bool(optimizer_config.get("adaptive_lr", True)),
        annealing=float(optimizer_config.get("annealing", 1.0)),
        max_time=float(max_time),
        correct_margin=float(optimizer_config.get("correct_margin", 0.0)),
        non_target_scale=float(optimizer_config.get("non_target_scale", 1.0)),
        update_scale=float(optimizer_config.get("update_scale", 1.0)),
        temporal_margin=float(optimizer_config.get("temporal_margin", 0.005)),
        temporal_scale_cap=float(optimizer_config.get("temporal_scale_cap", 0.02)),
        temporal_contrast_target_scale=float(
            optimizer_config.get("temporal_contrast_target_scale", 1.0)
        ),
        temporal_hard_negative_k=int(optimizer_config.get("temporal_hard_negative_k", 3)),
        temporal_target_prototypes=int(optimizer_config.get("temporal_target_prototypes", 1)),
        temporal_negative_prototypes=int(optimizer_config.get("temporal_negative_prototypes", 1)),
        temporal_prototype_decay=float(optimizer_config.get("temporal_prototype_decay", 0.5)),
        temporal_floor_scale=float(optimizer_config.get("temporal_floor_scale", 0.0)),
        temporal_stability_min_spikes=float(
            optimizer_config.get("temporal_stability_min_spikes", 45.0)
        ),
        temporal_guard_anti_scale=float(optimizer_config.get("temporal_guard_anti_scale", 0.25)),
        temporal_guard_target_scale=float(
            optimizer_config.get("temporal_guard_target_scale", 1.25)
        ),
        temporal_correct_update_scale=float(
            optimizer_config.get("temporal_correct_update_scale", 1.0)
        ),
        temporal_confident_correct_scale=(
            None
            if "temporal_confident_correct_scale" not in optimizer_config
            else float(optimizer_config["temporal_confident_correct_scale"])
        ),
        temporal_anti_update_scale=float(optimizer_config.get("temporal_anti_update_scale", 1.0)),
    )

    regularizer_config = config["regularizer"]
    if not bool(regularizer_config.get("use_two_thr", True)):
        raise ValueError("regularizer.use_two_thr must be true.")
    regularizer = CompetitionRegularizerTwoTorch(
        layer=layer,
        thr_lr=float(regularizer_config["thr_lr"]),
        thr_anneal=float(regularizer_config.get("thr_anneal", 1.0)),
    )

    return TemporalReadout(
        n_classes=n_classes,
        layer=layer,
        optimizer=optimizer,
        regularizer=regularizer,
        config=config["trainer"],
        output_dir=output_dir,
    )
