"""Load pre-generated simulation datasets.

Lightweight module with no sbibm dependency — safe to import on
GPU-only environments that don't have simulator packages installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_single_task(path: str | Path) -> dict:
    """Load a single-task dataset and return a dict matching simulate()'s format.

    The returned dict has all keys that simulate() returns except "task",
    which can be obtained via get_task(data["task_name"]).
    """
    d = np.load(str(path), allow_pickle=True)
    return {
        "thetas_train": d["thetas_train"],
        "xs_train": d["xs_train"],
        "thetas_val": d["thetas_val"],
        "xs_val": d["xs_val"],
        "dim_theta": int(d["dim_theta"]),
        "dim_x": int(d["dim_x"]),
        "task_name": str(d["task_name"]),
        "seed": int(d["seed"]),
    }


def load_multi_task(path: str | Path) -> list[dict]:
    """Load a multi-task dataset and return a list of per-task dicts.

    Each dict has the same format as load_single_task().
    """
    d = np.load(str(path), allow_pickle=True)
    n_tasks = int(d["n_tasks"])
    task_names = d["task_names"]
    dim_thetas = d["dim_thetas"]
    dim_xs = d["dim_xs"]
    seed = int(d["seed"])

    results = []
    for i in range(n_tasks):
        results.append({
            "thetas_train": d[f"thetas_train_{i}"],
            "xs_train": d[f"xs_train_{i}"],
            "thetas_val": d[f"thetas_val_{i}"],
            "xs_val": d[f"xs_val_{i}"],
            "dim_theta": int(dim_thetas[i]),
            "dim_x": int(dim_xs[i]),
            "task_name": str(task_names[i]),
            "seed": seed,
        })
    return results
