from __future__ import annotations

import math
from typing import Optional

import deepmimo as dm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _as_pos_2d(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values.reshape(1, -1)
    return values.reshape(-1, values.shape[-1])


def get_plot_dataset(bundle):
    dataset = bundle["dataset"]
    if isinstance(dataset, list):
        return dataset[0]
    try:
        nested = dataset["datasets"]
    except Exception:
        nested = None
    if nested is not None:
        return nested[0]
    return dataset


def resolve_raw_user_index(bundle, dataset_wrapper, sample_index, atol=1e-5):
    plot_dataset = get_plot_dataset(bundle)
    base_dataset = dataset_wrapper.base_dataset
    rx_pos = np.asarray(base_dataset.dataset_filtered["rx_pos"][sample_index], dtype=float)
    tx_pos = np.asarray(base_dataset.dataset_filtered["tx_pos"][sample_index], dtype=float)

    rx_all = _as_pos_2d(plot_dataset.rx_pos)
    tx_all = _as_pos_2d(plot_dataset.tx_pos)
    tx_ref = tx_all[0]

    tx_ok = np.all(np.isclose(tx_ref, tx_pos, atol=atol), axis=-1) if tx_all.ndim > 1 else np.allclose(tx_ref, tx_pos, atol=atol)
    if isinstance(tx_ok, np.ndarray):
        tx_ok = bool(np.any(tx_ok))
    if not tx_ok:
        raise ValueError("Could not match notebook sample TX position to the loaded DeepMIMO dataset.")

    deltas = np.linalg.norm(rx_all - rx_pos[None, :], axis=1)
    raw_user_idx = int(np.argmin(deltas))
    if not math.isfinite(float(deltas[raw_user_idx])) or float(deltas[raw_user_idx]) > 1e-3:
        raise ValueError("Could not reliably match notebook sample RX position to a DeepMIMO user index.")
    return raw_user_idx


def plot_rays_safe(ds, user_idx, ax=None, proj_3D=True, title=None):
    ax = dm.plot_rays(
        _as_pos_2d(ds.rx_pos)[user_idx],
        _as_pos_2d(ds.tx_pos)[0],
        np.asarray(ds.inter_pos)[user_idx],
        np.asarray(ds.inter)[user_idx],
        ax=ax,
        proj_3D=proj_3D,
    )
    if title is not None:
        ax.set_title(title)
    return ax


def _sorted_valid_raw_path_indices(dataset_wrapper, sample_index):
    base_dataset = dataset_wrapper.base_dataset
    sort_by = getattr(base_dataset, "sort_by", "power")

    if sort_by == "power":
        indices = np.argsort(-np.array(base_dataset.dataset_filtered["power"][sample_index]))
    elif sort_by == "delay":
        indices = np.argsort(np.array(base_dataset.dataset_filtered["delay"][sample_index]))
    else:
        raise ValueError(f"Unknown sort_by option: {sort_by}")

    valid_indices = []
    path_keys = ["delay", "power", "phase", "aoa_az", "aoa_el"]
    if getattr(base_dataset, "include_aod", False):
        path_keys.extend(["aod_az", "aod_el"])

    for raw_idx in indices.tolist():
        broken = False
        for key in path_keys:
            value = base_dataset.dataset_filtered[key][sample_index][raw_idx]
            if np.isnan(value):
                broken = True
                break
        if not broken:
            valid_indices.append(int(raw_idx))
    return valid_indices


def resolve_raw_path_index(dataset_wrapper, sample_index, query_path_index):
    valid_indices = _sorted_valid_raw_path_indices(dataset_wrapper, sample_index)
    if query_path_index < 1 or query_path_index > len(valid_indices):
        raise ValueError(
            f"query_path_index={query_path_index} is out of range for this sample; valid path count is {len(valid_indices)}."
        )
    return int(valid_indices[query_path_index - 1])


def plot_single_path_safe(ds, user_idx, raw_path_idx, ax=None, proj_3D=True, title=None):
    inter_pos_all = np.asarray(ds.inter_pos)[user_idx]
    inter_all = np.asarray(ds.inter)[user_idx]
    inter_pos_one = inter_pos_all[raw_path_idx : raw_path_idx + 1]
    inter_one = inter_all[raw_path_idx : raw_path_idx + 1]

    ax = dm.plot_rays(
        _as_pos_2d(ds.rx_pos)[user_idx],
        _as_pos_2d(ds.tx_pos)[0],
        inter_pos_one,
        inter_one,
        ax=ax,
        proj_3D=proj_3D,
    )
    if title is not None:
        ax.set_title(title)
    return ax


def plot_multipath_on_scene(user_idx, ds, title=None):
    ax = ds.scene.plot(title=False, proj_3D=False)
    plot_rays_safe(ds, user_idx, ax=ax, proj_3D=False)
    ax.set_title(title or f"User {user_idx}: multipath over environment")
    plt.tight_layout()
    return ax


def _decode_interaction_sequence(inter_code):
    try:
        if np.isnan(inter_code):
            return []
    except Exception:
        pass
    try:
        return [int(ch) for ch in str(int(inter_code))]
    except Exception:
        return []


def _valid_interaction_positions(path_inter_pos):
    arr = np.asarray(path_inter_pos, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    valid = np.isfinite(arr).all(axis=1)
    arr = arr[valid]
    if arr.size == 0:
        return arr
    keep = ~np.all(np.isclose(arr, 0.0, atol=1e-8), axis=1)
    return arr[keep]


def plot_selected_path_on_scene(ds, user_idx, raw_path_idx, ax=None, title=None):
    if ax is None:
        ax = ds.scene.plot(title=False, proj_3D=False)

    tx_pos = _as_pos_2d(ds.tx_pos)[0][:2]
    rx_pos = _as_pos_2d(ds.rx_pos)[user_idx][:2]
    path_inter_pos = np.asarray(ds.inter_pos)[user_idx][raw_path_idx]
    path_inter_code = np.asarray(ds.inter)[user_idx][raw_path_idx]

    inter_positions = _valid_interaction_positions(path_inter_pos)
    inter_types = _decode_interaction_sequence(path_inter_code)
    n = min(len(inter_positions), len(inter_types))
    inter_positions = inter_positions[:n, :2] if n > 0 else np.zeros((0, 2), dtype=float)
    inter_types = inter_types[:n]

    polyline = np.vstack([tx_pos[None, :], inter_positions, rx_pos[None, :]])
    ax.plot(polyline[:, 0], polyline[:, 1], color="black", linewidth=1.8, alpha=0.9, zorder=6)

    color_map = {
        1: ("Reflection", "red"),
        2: ("Diffraction", "orange"),
        3: ("Scattering", "blue"),
        4: ("Transmission", "purple"),
    }
    used_labels = set()
    for pos, inter_type in zip(inter_positions, inter_types):
        label, color = color_map.get(inter_type, ("Interaction", "black"))
        plot_label = label if label not in used_labels else None
        used_labels.add(label)
        ax.scatter(
            pos[0],
            pos[1],
            s=28,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            zorder=7,
            label=plot_label,
        )

    if title is not None:
        ax.set_title(title)
    return ax


def plot_spatial_attention_map_with_scene(
    object_meta_df: pd.DataFrame,
    attn_long_df: pd.DataFrame,
    bundle,
    dataset_wrapper,
    sample_index: int,
    layer_idx: int = -1,
    query_path_index: int = 1,
    overlay_rays: bool = True,
    show_tx_rx: bool = True,
    annotate_tokens: bool = True,
    figsize=(9, 7),
):
    if object_meta_df.empty or attn_long_df.empty:
        print("No object attention data available.")
        return

    available_layers = sorted(attn_long_df["layer"].unique())
    layer_idx = available_layers[layer_idx] if layer_idx < 0 else layer_idx

    plot_df = attn_long_df[
        (attn_long_df["layer"] == layer_idx) &
        (attn_long_df["query_path_index"] == query_path_index)
    ][["token_index", "attention"]].merge(object_meta_df, on="token_index", how="left")

    plot_dataset = get_plot_dataset(bundle)
    raw_user_idx = resolve_raw_user_index(bundle, dataset_wrapper, sample_index)
    raw_path_idx = resolve_raw_path_index(dataset_wrapper, sample_index, query_path_index)
    tx_pos = _as_pos_2d(plot_dataset.tx_pos)[0]
    rx_pos = _as_pos_2d(plot_dataset.rx_pos)[raw_user_idx]

    ax = plot_dataset .scene.plot(title=False, proj_3D=False)
    ax.figure.set_size_inches(*figsize)

    if overlay_rays:
        plot_selected_path_on_scene(plot_dataset, raw_user_idx, raw_path_idx, ax=ax)

    scatter = ax.scatter(
        plot_df["center_x"],
        plot_df["center_y"],
        s=80 + 900 * plot_df["attention"],
        c=plot_df["attention"],
        cmap="inferno",
        edgecolors="black",
        linewidths=0.4,
        alpha=0.95,
        zorder=5,
    )

    if annotate_tokens:
        for _, row in plot_df.iterrows():
            ax.text(row["center_x"], row["center_y"], str(int(row["token_index"])), fontsize=8, zorder=6)

    if show_tx_rx:
        ax.scatter(tx_pos[0], tx_pos[1], marker="^", s=140, color="#1f77b4", edgecolors="black", linewidths=0.6, zorder=7, label="TX")
        ax.scatter(rx_pos[0], rx_pos[1], marker="o", s=120, color="#2ca02c", edgecolors="black", linewidths=0.6, zorder=7, label="RX")
        ax.legend(loc="best")

    ax.figure.colorbar(scatter, ax=ax, label="Cross-attention weight")
    ax.set_title(
        f"Spatial attention over selected buildings | layer {layer_idx} | path step {query_path_index} | raw path {raw_path_idx}"
    )
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    plt.tight_layout()
    return ax
