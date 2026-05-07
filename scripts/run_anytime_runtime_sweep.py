"""Anytime/runtime SBI benchmark.

This benchmark separates three budgets that are otherwise easy to conflate:

* simulation time,
* method fit time,
* posterior query/sample time.

It writes one cached cell per task/method/seed/n_train and an aggregate CSV,
then plots C2ST against wall-clock and against simulation budget.
Evaluation time is deliberately excluded from method runtime.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfn_testing.sbi.density_estimators import (  # noqa: E402
    build_flow,
    get_flow_defaults,
    sample_posterior,
    train_flow,
)
from pfn_testing.sbi.sbi_baselines import (  # noqa: E402
    METHOD_REGISTRY,
    _get_sbi_class,
    _prior_to_device,
)
from pfn_testing.sbi.sbibm_utils import compute_c2st, get_task, simulate  # noqa: E402
from pfn_testing.sbi.tabpfn_npe import PCAReducer, TabPFNEmbedder  # noqa: E402
from scripts.layer_quantile_probe import _pinball_np  # noqa: E402

try:
    from npe_pfn import TabPFN_Based_NPE_PFN
except ImportError:  # pragma: no cover - optional dependency in some envs
    TabPFN_Based_NPE_PFN = None


OUT_DIR = Path("pfn_testing/sbi/outputs/anytime_runtime")
CELL_DIR = OUT_DIR / "cells"
FIG_DIR = OUT_DIR / "figures"
AGG_CSV = OUT_DIR / "anytime_runtime_summary.csv"

DEFAULT_TASKS = ["two_moons", "slcp", "ar1_ts_t50"]
DEFAULT_METHODS = ["pfn_npe_pca64", "npe_pfn", "sbi_npe", "bayesflow_nsf", "sbi_nle"]
SBI_METHODS = {"sbi_npe": "npe", "sbi_nle": "nle", "sbi_nre": "nre", "sbi_fmpe": "fmpe"}

METHOD_LABELS = {
    "pfn_npe_pca64": "PFN-NPE PCA64",
    "npe_pfn": "NPE-PFN",
    "sbi_npe": "sbi-NPE",
    "bayesflow_nsf": "BayesFlow NSF",
    "sbi_nle": "sbi-NLE",
    "sbi_nre": "sbi-NRE",
    "sbi_fmpe": "sbi-FMPE",
}
METHOD_COLORS = {
    "pfn_npe_pca64": "#0072B2",
    "npe_pfn": "#D55E00",
    "sbi_npe": "#009E73",
    "bayesflow_nsf": "#332288",
    "sbi_nle": "#CC79A7",
    "sbi_nre": "#E69F00",
    "sbi_fmpe": "#56B4E9",
}
TASK_LABELS = {
    "two_moons": "Two moons",
    "slcp": "SLCP",
    "ar1_ts_t50": "AR(1), T=50",
}
TAUS = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float32)


@dataclass
class FlowTrainConfig:
    lr: float
    batch_size: int
    max_epochs: int
    patience: int
    grad_clip: float = 5.0
    flow_type: str = "nsf"


@dataclass
class CellConfig:
    task: str
    method: str
    seed: int
    n_train: int
    n_val: int
    n_ref: int
    n_posterior_samples: int
    device: str
    context_size: int
    embed_dim: int
    flow_max_epochs: int
    flow_patience: int
    flow_batch_size: int
    flow_lr: float
    sbi_max_num_epochs: int
    sbi_stop_after_epochs: int
    sbi_batch_size: int
    sbi_lr: float
    sbi_density_estimator: str | None
    sbi_sample_with: str
    sbi_mcmc_method: str
    sbi_vi_max_num_iters: int
    sbi_vi_min_num_iters: int
    sbi_vi_lr: float
    sbi_vi_quality_control: bool
    bayesflow_max_epochs: int
    bayesflow_patience: int
    bayesflow_batch_size: int
    bayesflow_lr: float
    bayesflow_depth: int | None
    bayesflow_widths: list[int]
    bayesflow_bins: int
    c2st_folds: int
    c2st_max_epochs: int
    c2st_seed: int
    npe_pfn_filter_context_size: int
    npe_pfn_n_estimators: int


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def safe_method(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", method)


def cell_stem(cfg: CellConfig) -> str:
    stem = (
        f"{cfg.task}_s{cfg.seed}_n{cfg.n_train}_{safe_method(cfg.method)}_"
        f"obs{cfg.n_ref}_ps{cfg.n_posterior_samples}"
    )
    if cfg.method == "pfn_npe_pca64":
        if cfg.context_size != 1_000:
            stem += f"_ctx{cfg.context_size}"
        if cfg.embed_dim != 64:
            stem += f"_pca{cfg.embed_dim}"
    if cfg.method == "npe_pfn":
        if cfg.npe_pfn_n_estimators != 1:
            stem += f"_est{cfg.npe_pfn_n_estimators}"
        if cfg.npe_pfn_filter_context_size != 10_000:
            stem += f"_fctx{cfg.npe_pfn_filter_context_size}"
    if cfg.method in SBI_METHODS:
        sbi_method = SBI_METHODS[cfg.method]
        method_info = METHOD_REGISTRY[sbi_method]
        default_estimator = method_info["default_estimator"]
        if cfg.sbi_density_estimator and cfg.sbi_density_estimator != default_estimator:
            stem += f"_de{safe_method(cfg.sbi_density_estimator)}"
        if not method_info["amortized"]:
            if cfg.sbi_sample_with != "mcmc":
                stem += f"_sample{safe_method(cfg.sbi_sample_with)}"
                if cfg.sbi_sample_with == "vi":
                    stem += f"_vi{cfg.sbi_vi_max_num_iters}"
            elif cfg.sbi_mcmc_method != "slice_np_vectorized":
                stem += f"_mcmc{safe_method(cfg.sbi_mcmc_method)}"
    if cfg.method == "bayesflow_nsf":
        if cfg.bayesflow_depth is not None:
            stem += f"_d{cfg.bayesflow_depth}"
        if cfg.bayesflow_widths:
            stem += "_w" + "-".join(str(width) for width in cfg.bayesflow_widths)
        if cfg.bayesflow_bins != 16:
            stem += f"_bins{cfg.bayesflow_bins}"
        if cfg.bayesflow_max_epochs != 200:
            stem += f"_ep{cfg.bayesflow_max_epochs}"
        if cfg.bayesflow_patience != 20:
            stem += f"_pat{cfg.bayesflow_patience}"
        if cfg.bayesflow_batch_size != 256:
            stem += f"_bs{cfg.bayesflow_batch_size}"
        if cfg.bayesflow_lr != 5e-4:
            stem += f"_lr{cfg.bayesflow_lr:g}"
    return stem


def cell_paths(cfg: CellConfig) -> tuple[Path, Path]:
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    stem = cell_stem(cfg)
    return CELL_DIR / f"{stem}.json", CELL_DIR / f"{stem}.npz"


def timer() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def cleanup_torch() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def module_devices(module: object) -> list[str]:
    """Return unique parameter/buffer devices for a torch module."""
    if not isinstance(module, torch.nn.Module):
        return []
    devices = {str(param.device) for param in module.parameters()}
    devices.update(str(buffer.device) for buffer in module.buffers())
    return sorted(devices)


def tabpfn_embedder_devices(embedder: TabPFNEmbedder) -> list[str]:
    try:
        return module_devices(embedder._get_transformer_model())
    except Exception:
        return []


def prior_reference(task_name: str, n_ref: int) -> tuple[object, np.ndarray, np.ndarray]:
    task = get_task(task_name)
    x_ref = np.stack([
        task.get_observation(num_observation=i + 1).numpy().reshape(-1)
        for i in range(n_ref)
    ]).astype(np.float32)
    theta_true = np.stack([
        task.get_true_parameters(num_observation=i + 1).numpy().reshape(-1)
        for i in range(n_ref)
    ]).astype(np.float32)
    return task, x_ref, theta_true


def metric_decomposition(
    samples: np.ndarray,
    ref: np.ndarray,
    folds: int,
    max_epochs: int,
    seed: int,
) -> dict[str, float]:
    n = min(len(samples), len(ref))
    a = np.asarray(samples[:n], dtype=np.float64)
    b = np.asarray(ref[:n], dtype=np.float64)
    joint = compute_c2st(a, b, n_folds=folds, max_epochs=max_epochs, seed=seed)
    marginal = float(np.mean([
        compute_c2st(
            a[:, d:d + 1],
            b[:, d:d + 1],
            n_folds=folds,
            max_epochs=max_epochs,
            seed=seed,
        )
        for d in range(a.shape[1])
    ]))
    ar, br = pooled_rank_transform(a, b)
    rank = compute_c2st(ar, br, n_folds=folds, max_epochs=max_epochs, seed=seed)
    return {"joint_c2st": joint, "marginal_c2st": marginal, "rank_c2st": rank}


def pooled_rank_transform(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    a_out = np.empty_like(a, dtype=np.float64)
    b_out = np.empty_like(b, dtype=np.float64)
    n_a = a.shape[0]
    n_b = b.shape[0]
    for dim in range(a.shape[1]):
        pooled = np.concatenate([a[:, dim], b[:, dim]])
        order = np.argsort(pooled, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(pooled) + 1)
        ranks = ranks / (len(pooled) + 1)
        a_out[:, dim] = ranks[:n_a]
        b_out[:, dim] = ranks[n_a:n_a + n_b]
    return a_out, b_out


def evaluate_samples(
    task: object,
    samples: np.ndarray,
    theta_true: np.ndarray,
    cfg: CellConfig,
) -> dict[str, object]:
    n_ref = samples.shape[0]
    rows = []
    flow_q = np.zeros((n_ref, len(TAUS), samples.shape[2]), dtype=np.float32)
    emp_q = np.zeros_like(flow_q)
    for obs_zero in range(n_ref):
        obs = obs_zero + 1
        ref = task.get_reference_posterior_samples(num_observation=obs).numpy()
        ref = ref[: cfg.n_posterior_samples]
        metrics = metric_decomposition(
            samples[obs_zero],
            ref,
            cfg.c2st_folds,
            cfg.c2st_max_epochs,
            cfg.c2st_seed,
        )
        metrics["obs"] = obs
        rows.append(metrics)
        flow_q[obs_zero] = np.quantile(samples[obs_zero], TAUS, axis=0)
        emp_q[obs_zero] = np.quantile(ref, TAUS, axis=0)

    pinball = float(_pinball_np(theta_true, flow_q.transpose(1, 0, 2), TAUS).mean())
    return {
        "per_obs": rows,
        "joint_c2st": float(np.mean([r["joint_c2st"] for r in rows])),
        "marginal_c2st": float(np.mean([r["marginal_c2st"] for r in rows])),
        "rank_c2st": float(np.mean([r["rank_c2st"] for r in rows])),
        "pinball": pinball,
    }


def train_sample_pfn_npe(
    data: dict[str, object],
    x_ref: np.ndarray,
    cfg: CellConfig,
) -> tuple[np.ndarray, float, list[float], dict[str, object]]:
    dim_theta = int(data["dim_theta"])
    emb_device = cfg.device
    if emb_device == "auto":
        emb_device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = timer()
    embedder = TabPFNEmbedder(
        context_size=min(cfg.context_size, cfg.n_train),
        seed=cfg.seed,
        label_strategy="per_dim",
        layer=None,
        model_type="regressor",
        device=emb_device,
    )
    embedder.fit(data["xs_train"], thetas=data["thetas_train"])
    emb_train = embedder.transform(data["xs_train"])
    emb_val = embedder.transform(data["xs_val"])
    e_ref = embedder.transform(x_ref)
    reducer = PCAReducer(min(cfg.embed_dim, emb_train.shape[1], emb_train.shape[0]))
    emb_train = reducer.fit_transform(emb_train)
    emb_val = reducer.transform(emb_val)
    e_ref = reducer.transform(e_ref)

    defaults = get_flow_defaults(dim_theta)
    flow = build_flow(
        dim_theta=dim_theta,
        dim_context=emb_train.shape[1],
        n_transforms=defaults["n_transforms"],
        hidden_features=defaults["hidden_features"],
        n_bins=8,
        flow_type="nsf",
    )
    flow_cfg = FlowTrainConfig(
        lr=cfg.flow_lr,
        batch_size=cfg.flow_batch_size,
        max_epochs=cfg.flow_max_epochs,
        patience=cfg.flow_patience,
    )
    history = train_flow(
        flow,
        data["thetas_train"],
        emb_train,
        data["thetas_val"],
        emb_val,
        flow_cfg,
    )
    train_seconds = timer() - t0

    sample_times = []
    samples = np.zeros((cfg.n_ref, cfg.n_posterior_samples, dim_theta), dtype=np.float32)
    for obs_zero in range(cfg.n_ref):
        t_obs = timer()
        samples[obs_zero] = sample_posterior(
            flow,
            e_ref[obs_zero],
            history["theta_mean"],
            history["theta_std"],
            cfg.n_posterior_samples,
        ).astype(np.float32)
        sample_times.append(timer() - t_obs)
    meta = {
        "embedding_dim": int(emb_train.shape[1]),
        "best_epoch": int(history["best_epoch"]),
        "tabpfn_embedder_device": emb_device,
        "tabpfn_embedder_model_devices": tabpfn_embedder_devices(embedder),
        "flow_devices": module_devices(flow),
    }
    return samples, train_seconds, sample_times, meta


def train_sample_npe_pfn(
    data: dict[str, object],
    x_ref: np.ndarray,
    cfg: CellConfig,
) -> tuple[np.ndarray, float, list[float], dict[str, object]]:
    if TabPFN_Based_NPE_PFN is None:
        raise ImportError("npe_pfn is not installed")
    task = data["task"]
    dim_theta = int(data["dim_theta"])
    model_path = Path.home() / ".cache/tabpfn/tabpfn-v2-regressor.ckpt"
    if not model_path.exists():
        raise FileNotFoundError(
            "Cached TabPFN v2 regressor checkpoint is missing: "
            f"{model_path}. Run the PFN-NPE embedder once or provide the "
            "checkpoint before benchmarking NPE-PFN."
        )
    model_device = "cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu"

    t0 = timer()
    estimator = TabPFN_Based_NPE_PFN(
        prior=task.get_prior_dist(),
        filter_context_size=min(cfg.npe_pfn_filter_context_size, cfg.n_train),
        show_progress_bars=False,
        regressor_init_kwargs={
            "model_path": model_path,
            "n_estimators": cfg.npe_pfn_n_estimators,
            "device": model_device,
        },
    )
    estimator.append_simulations(
        torch.tensor(data["thetas_train"], dtype=torch.float32),
        torch.tensor(data["xs_train"], dtype=torch.float32),
    )
    train_seconds = timer() - t0

    sample_times = []
    samples = np.zeros((cfg.n_ref, cfg.n_posterior_samples, dim_theta), dtype=np.float32)
    for obs_zero in range(cfg.n_ref):
        t_obs = timer()
        x_obs_t = torch.tensor(x_ref[obs_zero].reshape(1, -1), dtype=torch.float32)
        s = estimator.sample(
            sample_shape=torch.Size([cfg.n_posterior_samples]),
            x=x_obs_t,
        )
        samples[obs_zero] = s.detach().cpu().numpy().astype(np.float32)
        sample_times.append(timer() - t_obs)
    return samples, train_seconds, sample_times, {
        "filter_context_size": min(cfg.npe_pfn_filter_context_size, cfg.n_train),
        "n_estimators": cfg.npe_pfn_n_estimators,
        "tabpfn_model_path": str(model_path),
        "tabpfn_device": model_device,
    }


def train_sample_sbi_method(
    data: dict[str, object],
    x_ref: np.ndarray,
    cfg: CellConfig,
) -> tuple[np.ndarray, float, list[float], dict[str, object]]:
    sbi_method = SBI_METHODS[cfg.method]
    method_info = METHOD_REGISTRY[sbi_method]
    sbi_class = _get_sbi_class(sbi_method)
    device = "cuda" if torch.cuda.is_available() and cfg.device != "cpu" else "cpu"
    prior = _prior_to_device(data["task"].get_prior_dist(), device)
    estimator = cfg.sbi_density_estimator or method_info["default_estimator"]
    estimator_arg: str | Callable = estimator
    if sbi_method == "fmpe" and device != "cpu":
        from sbi.neural_nets import flowmatching_nn

        build_flowmatcher = flowmatching_nn(model=estimator)

        def build_cuda_flowmatcher(batch_theta: torch.Tensor, batch_x: torch.Tensor) -> torch.nn.Module:
            net = build_flowmatcher(batch_theta, batch_x)
            transform = getattr(net, "zscore_transform_input", None)
            for attr in ("loc", "scale"):
                value = getattr(transform, attr, None)
                if isinstance(value, torch.Tensor):
                    setattr(transform, attr, value.to(device))
            return net

        estimator_arg = build_cuda_flowmatcher
    init_kwargs = {
        "prior": prior,
        method_info["estimator_kwarg"]: estimator_arg,
        "device": device,
    }

    t0 = timer()
    inference = sbi_class(**init_kwargs)
    theta_train_t = torch.tensor(data["thetas_train"], dtype=torch.float32, device=device)
    x_train_t = torch.tensor(data["xs_train"], dtype=torch.float32, device=device)
    inference.append_simulations(
        theta_train_t,
        x_train_t,
    )
    train_kwargs = {
        "training_batch_size": cfg.sbi_batch_size,
        "stop_after_epochs": cfg.sbi_stop_after_epochs,
        "max_num_epochs": cfg.sbi_max_num_epochs,
        "learning_rate": cfg.sbi_lr,
    }
    trained_net = inference.train(**train_kwargs)
    build_kwargs: dict[str, object] = {}
    if not method_info["amortized"]:
        build_kwargs["sample_with"] = cfg.sbi_sample_with
        if cfg.sbi_sample_with == "mcmc":
            build_kwargs["mcmc_method"] = cfg.sbi_mcmc_method
    use_vi = (not method_info["amortized"]) and cfg.sbi_sample_with == "vi"
    posterior = None if use_vi else inference.build_posterior(**build_kwargs)
    train_seconds = timer() - t0

    dim_theta = int(data["dim_theta"])
    sample_times = []
    samples = np.zeros((cfg.n_ref, cfg.n_posterior_samples, dim_theta), dtype=np.float32)
    for obs_zero in range(cfg.n_ref):
        t_obs = timer()
        x_obs_t = torch.tensor(x_ref[obs_zero], dtype=torch.float32, device=device)
        if use_vi:
            obs_posterior = inference.build_posterior(**build_kwargs)
            obs_posterior.train(
                x=x_obs_t,
                max_num_iters=cfg.sbi_vi_max_num_iters,
                min_num_iters=cfg.sbi_vi_min_num_iters,
                learning_rate=cfg.sbi_vi_lr,
                show_progress_bar=False,
                quality_control=cfg.sbi_vi_quality_control,
            )
            s = obs_posterior.sample((cfg.n_posterior_samples,), x=x_obs_t)
        else:
            assert posterior is not None
            s = posterior.sample((cfg.n_posterior_samples,), x=x_obs_t)
        samples[obs_zero] = s.detach().cpu().numpy().astype(np.float32)
        sample_times.append(timer() - t_obs)
    return samples, train_seconds, sample_times, {
        "sbi_method": sbi_method,
        "estimator": estimator,
        "sbi_device": device,
        "sbi_train_tensor_devices": sorted({str(theta_train_t.device), str(x_train_t.device)}),
        "sbi_trained_net_devices": module_devices(trained_net),
        "sbi_sample_with": cfg.sbi_sample_with if not method_info["amortized"] else None,
        "sbi_mcmc_method": cfg.sbi_mcmc_method if cfg.sbi_sample_with == "mcmc" else None,
        "sbi_vi_max_num_iters": cfg.sbi_vi_max_num_iters if use_vi else None,
        "sbi_vi_min_num_iters": cfg.sbi_vi_min_num_iters if use_vi else None,
        "sbi_vi_lr": cfg.sbi_vi_lr if use_vi else None,
        "sbi_vi_quality_control": cfg.sbi_vi_quality_control if use_vi else None,
    }


def get_bayesflow_defaults(dim_theta: int) -> dict[str, object]:
    """Dimension-aware BayesFlow defaults for the stronger NSF comparator."""
    if dim_theta <= 5:
        return {"depth": 6, "widths": [128, 128]}
    return {"depth": 8, "widths": [256, 256]}


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _keras_weight_devices(model: object) -> list[str]:
    devices: list[str] = []
    for weight in getattr(model, "trainable_weights", []):
        value = getattr(weight, "value", None)
        if value is not None and hasattr(value, "device"):
            devices.append(str(value.device))
    return sorted(set(devices))


def train_sample_bayesflow_nsf(
    data: dict[str, object],
    x_ref: np.ndarray,
    cfg: CellConfig,
) -> tuple[np.ndarray, float, list[float], dict[str, object]]:
    import bayesflow as bf
    import keras
    from keras.src.backend.torch.core import device_scope, get_device

    keras.utils.set_random_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dim_theta = int(data["dim_theta"])
    defaults = get_bayesflow_defaults(dim_theta)
    depth = cfg.bayesflow_depth if cfg.bayesflow_depth is not None else int(defaults["depth"])
    widths = cfg.bayesflow_widths or list(defaults["widths"])

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BayesFlow was requested on CUDA but torch.cuda.is_available() is false.")
    bayesflow_device = "cuda" if cfg.device != "cpu" and torch.cuda.is_available() else "cpu"
    old_default_device = torch.get_default_device()
    if bayesflow_device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.set_default_device(bayesflow_device)

    try:
        with device_scope(bayesflow_device):
            adapter = bf.Adapter().convert_dtype("float64", "float32")
            inference_network = bf.networks.CouplingFlow(
                depth=depth,
                transform="spline",
                permutation="random",
                use_actnorm=True,
                subnet_kwargs={"widths": widths},
                transform_kwargs={"bins": cfg.bayesflow_bins},
            )
            approximator = bf.ContinuousApproximator(
                adapter=adapter,
                inference_network=inference_network,
                standardize="all",
            )

            train_data = {
                "inference_variables": np.asarray(data["thetas_train"], dtype=np.float32),
                "inference_conditions": np.asarray(data["xs_train"], dtype=np.float32),
            }
            val_data = {
                "inference_variables": np.asarray(data["thetas_val"], dtype=np.float32),
                "inference_conditions": np.asarray(data["xs_val"], dtype=np.float32),
            }

            train_dataset = bf.OfflineDataset(
                data=train_data,
                batch_size=cfg.bayesflow_batch_size,
                adapter=approximator.adapter,
                shuffle=True,
            )
            val_dataset = bf.OfflineDataset(
                data=val_data,
                batch_size=cfg.bayesflow_batch_size,
                adapter=approximator.adapter,
                shuffle=False,
            )

            approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.bayesflow_lr))
            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.bayesflow_patience,
                restore_best_weights=True,
                verbose=0,
            )

            t0 = timer()
            with torch.enable_grad():
                history = approximator.fit(
                    dataset=train_dataset,
                    epochs=cfg.bayesflow_max_epochs,
                    validation_data=val_dataset,
                    callbacks=[early_stop],
                    verbose=0,
                )
            train_seconds = timer() - t0

            weight_devices = _keras_weight_devices(approximator)
            if bayesflow_device == "cuda" and not all(device.startswith("cuda") for device in weight_devices):
                raise RuntimeError(
                    "BayesFlow trainable weights are not on CUDA: "
                    f"{weight_devices or ['<none found>']}"
                )

            val_loss = history.history.get("val_loss", [])
            best_epoch = int(np.argmin(val_loss)) if val_loss else len(history.history.get("loss", [])) - 1
            epochs_trained = int(len(history.history.get("loss", [])))

            sample_times = []
            samples = np.zeros((cfg.n_ref, cfg.n_posterior_samples, dim_theta), dtype=np.float32)
            for obs_zero in range(cfg.n_ref):
                conditions = {
                    "inference_conditions": np.asarray(x_ref[obs_zero].reshape(1, -1), dtype=np.float32),
                }
                t_obs = timer()
                result = approximator.sample(num_samples=cfg.n_posterior_samples, conditions=conditions)
                arr = _as_numpy(result["inference_variables"])
                samples[obs_zero] = arr[0].astype(np.float32)
                sample_times.append(timer() - t_obs)

        peak_memory_mb = (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if bayesflow_device == "cuda"
            else 0.0
        )
        return samples, train_seconds, sample_times, {
            "bayesflow_version": getattr(bf, "__version__", "unknown"),
            "bayesflow_transform": "spline",
            "bayesflow_device": bayesflow_device,
            "bayesflow_keras_device": str(get_device()),
            "bayesflow_weight_devices": weight_devices,
            "bayesflow_peak_memory_mb": peak_memory_mb,
            "bayesflow_depth": depth,
            "bayesflow_widths": widths,
            "bayesflow_bins": cfg.bayesflow_bins,
            "bayesflow_standardize": "all",
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
        }
    finally:
        torch.set_default_device(old_default_device)


METHOD_RUNNERS: dict[
    str,
    Callable[[dict[str, object], np.ndarray, CellConfig], tuple[np.ndarray, float, list[float], dict[str, object]]],
] = {
    "pfn_npe_pca64": train_sample_pfn_npe,
    "npe_pfn": train_sample_npe_pfn,
    "sbi_npe": train_sample_sbi_method,
    "bayesflow_nsf": train_sample_bayesflow_nsf,
    "sbi_nle": train_sample_sbi_method,
    "sbi_nre": train_sample_sbi_method,
    "sbi_fmpe": train_sample_sbi_method,
}


def run_cell(cfg: CellConfig, *, force: bool = False) -> dict[str, object]:
    json_path, npz_path = cell_paths(cfg)
    if json_path.exists() and npz_path.exists() and not force:
        with json_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["cached"] = True
        return row

    if cfg.method not in METHOD_RUNNERS:
        raise ValueError(f"Unknown method: {cfg.method}")

    print(
        f"\n[cell] task={cfg.task} method={cfg.method} seed={cfg.seed} "
        f"n_train={cfg.n_train}"
    )
    t_sim = timer()
    data = simulate(cfg.task, cfg.n_train, cfg.n_val, cfg.seed)
    simulation_seconds = timer() - t_sim
    task, x_ref, theta_true = prior_reference(cfg.task, cfg.n_ref)
    data["task"] = task

    try:
        samples, train_seconds, sample_times, meta = METHOD_RUNNERS[cfg.method](data, x_ref, cfg)
        eval_meta = evaluate_samples(task, samples, theta_true, cfg)
        sample_seconds_n_ref = float(np.sum(sample_times))
        row: dict[str, object] = {
            **asdict(cfg),
            "status": "ok",
            "simulation_seconds": float(simulation_seconds),
            "train_seconds": float(train_seconds),
            "sample_seconds_n_ref": sample_seconds_n_ref,
            "sample_seconds_per_obs_mean": float(np.mean(sample_times)),
            "sample_seconds_per_obs_std": float(np.std(sample_times)),
            "total_seconds_1_obs": float(simulation_seconds + train_seconds + sample_times[0]),
            "total_seconds_n_ref": float(simulation_seconds + train_seconds + sample_seconds_n_ref),
            "joint_c2st": eval_meta["joint_c2st"],
            "marginal_c2st": eval_meta["marginal_c2st"],
            "rank_c2st": eval_meta["rank_c2st"],
            "pinball": eval_meta["pinball"],
            "per_obs": eval_meta["per_obs"],
            "method_meta": meta,
            "cached": False,
        }
        np.savez_compressed(
            npz_path,
            samples=samples.astype(np.float32),
            x_ref=x_ref.astype(np.float32),
            theta_true=theta_true.astype(np.float32),
            task=cfg.task,
            method=cfg.method,
            seed=cfg.seed,
            n_train=cfg.n_train,
        )
    except Exception as exc:
        row = {
            **asdict(cfg),
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "simulation_seconds": float(simulation_seconds),
            "train_seconds": np.nan,
            "sample_seconds_n_ref": np.nan,
            "sample_seconds_per_obs_mean": np.nan,
            "sample_seconds_per_obs_std": np.nan,
            "total_seconds_1_obs": np.nan,
            "total_seconds_n_ref": np.nan,
            "joint_c2st": np.nan,
            "marginal_c2st": np.nan,
            "rank_c2st": np.nan,
            "pinball": np.nan,
            "per_obs": [],
            "method_meta": {},
            "cached": False,
        }
        print(f"  [error] {type(exc).__name__}: {exc}")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, default=str)
    cleanup_torch()
    return row


def flatten_row(row: dict[str, object]) -> dict[str, object]:
    skip = {"per_obs", "method_meta"}
    flat = {k: v for k, v in row.items() if k not in skip}
    meta = row.get("method_meta", {})
    if isinstance(meta, dict):
        for k, v in meta.items():
            flat[f"meta_{k}"] = v
    return flat


def write_aggregate(rows: list[dict[str, object]], out_csv: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat = [flatten_row(row) for row in rows]
    fields = sorted({key for row in flat for key in row})
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    print(f"Wrote {out_csv} ({len(rows)} rows)")


def load_all_cells() -> list[dict[str, object]]:
    rows = []
    if not CELL_DIR.exists():
        return rows
    for path in sorted(CELL_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def cell_variant_key(row: dict[str, object]) -> tuple[object, ...]:
    key: tuple[object, ...] = (
        row["task"],
        row["method"],
        int(row["seed"]),
        int(row["n_train"]),
        int(row["n_ref"]),
        int(row["n_posterior_samples"]),
    )
    if row.get("method") == "bayesflow_nsf":
        key += (
            row.get("bayesflow_depth"),
            tuple(row.get("bayesflow_widths") or []),
            row.get("bayesflow_bins"),
            row.get("bayesflow_max_epochs"),
            row.get("bayesflow_patience"),
            row.get("bayesflow_batch_size"),
            row.get("bayesflow_lr"),
        )
    return key


def aggregate_means(rows: list[dict[str, object]]) -> dict[tuple[str, str, int], dict[str, float]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row["task"]), str(row["method"]), int(row["n_train"]))
        grouped.setdefault(key, []).append(row)

    out: dict[tuple[str, str, int], dict[str, float]] = {}
    metrics = [
        "joint_c2st",
        "marginal_c2st",
        "rank_c2st",
        "pinball",
        "total_seconds_1_obs",
        "total_seconds_n_ref",
        "simulation_seconds",
        "train_seconds",
        "sample_seconds_n_ref",
        "sample_seconds_per_obs_mean",
    ]
    for key, values in grouped.items():
        summary = {"n": float(len(values))}
        for metric in metrics:
            arr = np.asarray([float(v[metric]) for v in values], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(arr))
            summary[f"{metric}_std"] = float(np.nanstd(arr))
        out[key] = summary
    return out


def plot_results(rows: list[dict[str, object]], out_dir: Path, formats: tuple[str, ...]) -> None:
    configure_style()
    summaries = aggregate_means(rows)
    if not summaries:
        print("No successful rows to plot.")
        return

    tasks = sorted(
        {key[0] for key in summaries},
        key=lambda x: (0, DEFAULT_TASKS.index(x)) if x in DEFAULT_TASKS else (1, x),
    )
    methods = sorted(
        {key[1] for key in summaries},
        key=lambda x: (0, DEFAULT_METHODS.index(x)) if x in DEFAULT_METHODS else (1, x),
    )

    fig, axes = plt.subplots(2, len(tasks), figsize=(4.0 * len(tasks), 5.8), squeeze=False)
    for col, task in enumerate(tasks):
        ax_time = axes[0, col]
        ax_budget = axes[1, col]
        for method in methods:
            pts = [
                (n_train, values)
                for (task_key, method_key, n_train), values in summaries.items()
                if task_key == task and method_key == method
            ]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            budgets = np.asarray([p[0] for p in pts], dtype=float)
            c2st = np.asarray([p[1]["joint_c2st_mean"] for p in pts], dtype=float)
            c2st_std = np.asarray([p[1]["joint_c2st_std"] for p in pts], dtype=float)
            runtime = np.asarray([p[1]["total_seconds_n_ref_mean"] for p in pts], dtype=float)
            runtime_std = np.asarray([p[1]["total_seconds_n_ref_std"] for p in pts], dtype=float)
            label = METHOD_LABELS.get(method, method)
            color = METHOD_COLORS.get(method, "0.3")
            ax_time.errorbar(
                runtime,
                c2st,
                xerr=runtime_std if len(pts) > 1 else None,
                yerr=c2st_std if len(pts) > 1 else None,
                marker="o",
                color=color,
                lw=1.5,
                capsize=2,
                label=label,
            )
            for x_val, y_val, budget in zip(runtime, c2st, budgets, strict=True):
                ax_time.text(x_val, y_val + 0.008, f"{int(budget)}", fontsize=6, ha="center")
            ax_budget.errorbar(
                budgets,
                c2st,
                yerr=c2st_std if len(pts) > 1 else None,
                marker="o",
                color=color,
                lw=1.5,
                capsize=2,
                label=label,
            )

        ax_time.axhline(0.5, color="0.45", ls=":", lw=0.9)
        ax_time.set_xscale("log")
        ax_time.set_title(TASK_LABELS.get(task, task))
        ax_time.set_xlabel("Total wall-clock for sampled observations (s)")
        ax_time.set_ylabel("Joint C2ST")
        ax_time.set_ylim(0.48, 1.02)

        ax_budget.axhline(0.5, color="0.45", ls=":", lw=0.9)
        ax_budget.set_xscale("log")
        ax_budget.set_xlabel("Training simulations")
        ax_budget.set_ylabel("Joint C2ST")
        ax_budget.set_ylim(0.48, 1.02)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle("Anytime SBI comparison: quality vs runtime and simulation budget", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"anytime_runtime_comparison.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(5.0, 1.2 * len(methods) * len(tasks)), 3.2))
    width = 0.8 / max(1, len(methods))
    task_offsets = np.arange(len(tasks))
    for idx, method in enumerate(methods):
        vals = []
        sems = []
        for task in tasks:
            arr = [
                values["sample_seconds_per_obs_mean_mean"]
                for (task_key, method_key, _), values in summaries.items()
                if task_key == task and method_key == method
            ]
            vals.append(float(np.nanmean(arr)) if arr else np.nan)
            sems.append(float(np.nanstd(arr)) if len(arr) > 1 else 0.0)
        ax.bar(
            task_offsets + (idx - (len(methods) - 1) / 2) * width,
            vals,
            width=width,
            yerr=sems,
            color=METHOD_COLORS.get(method, "0.3"),
            edgecolor="0.25",
            linewidth=0.5,
            capsize=2,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xticks(task_offsets)
    ax.set_xticklabels([TASK_LABELS.get(task, task) for task in tasks], rotation=18, ha="right")
    ax.set_ylabel("Posterior sample time per observation (s)")
    ax.set_yscale("log")
    ax.legend(frameon=False, ncol=min(len(methods), 4))
    fig.tight_layout()
    for fmt in formats:
        path = out_dir / f"anytime_query_time.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--budgets", nargs="+", type=int, default=[500, 1000, 3000, 10000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-ref", type=int, default=10)
    ap.add_argument("--n-posterior-samples", type=int, default=1000)
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--context-size", type=int, default=1000)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--flow-max-epochs", type=int, default=200)
    ap.add_argument("--flow-patience", type=int, default=20)
    ap.add_argument("--flow-batch-size", type=int, default=256)
    ap.add_argument("--flow-lr", type=float, default=5e-4)
    ap.add_argument("--sbi-max-num-epochs", type=int, default=200)
    ap.add_argument("--sbi-stop-after-epochs", type=int, default=20)
    ap.add_argument("--sbi-batch-size", type=int, default=200)
    ap.add_argument("--sbi-lr", type=float, default=5e-4)
    ap.add_argument("--sbi-density-estimator", default=None)
    ap.add_argument("--sbi-sample-with", default="mcmc")
    ap.add_argument("--sbi-mcmc-method", default="slice_np_vectorized")
    ap.add_argument("--sbi-vi-max-num-iters", type=int, default=500)
    ap.add_argument("--sbi-vi-min-num-iters", type=int, default=50)
    ap.add_argument("--sbi-vi-lr", type=float, default=1e-3)
    ap.add_argument("--sbi-vi-quality-control", action="store_true")
    ap.add_argument("--bayesflow-max-epochs", type=int, default=200)
    ap.add_argument("--bayesflow-patience", type=int, default=20)
    ap.add_argument("--bayesflow-batch-size", type=int, default=256)
    ap.add_argument("--bayesflow-lr", type=float, default=5e-4)
    ap.add_argument("--bayesflow-depth", type=int, default=None)
    ap.add_argument("--bayesflow-widths", nargs="+", type=int, default=None)
    ap.add_argument("--bayesflow-bins", type=int, default=16)
    ap.add_argument("--c2st-folds", type=int, default=5)
    ap.add_argument("--c2st-max-epochs", type=int, default=500)
    ap.add_argument("--c2st-seed", type=int, default=1)
    ap.add_argument("--npe-pfn-filter-context-size", type=int, default=10000)
    ap.add_argument("--npe-pfn-n-estimators", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--out-csv", type=Path, default=AGG_CSV)
    ap.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    return ap.parse_args()


def cfg_from_args(args: argparse.Namespace, task: str, method: str, seed: int, n_train: int) -> CellConfig:
    return CellConfig(
        task=task,
        method=method,
        seed=seed,
        n_train=n_train,
        n_val=args.n_val,
        n_ref=args.n_ref,
        n_posterior_samples=args.n_posterior_samples,
        device=args.device,
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        flow_max_epochs=args.flow_max_epochs,
        flow_patience=args.flow_patience,
        flow_batch_size=args.flow_batch_size,
        flow_lr=args.flow_lr,
        sbi_max_num_epochs=args.sbi_max_num_epochs,
        sbi_stop_after_epochs=args.sbi_stop_after_epochs,
        sbi_batch_size=args.sbi_batch_size,
        sbi_lr=args.sbi_lr,
        sbi_density_estimator=args.sbi_density_estimator,
        sbi_sample_with=args.sbi_sample_with,
        sbi_mcmc_method=args.sbi_mcmc_method,
        sbi_vi_max_num_iters=args.sbi_vi_max_num_iters,
        sbi_vi_min_num_iters=args.sbi_vi_min_num_iters,
        sbi_vi_lr=args.sbi_vi_lr,
        sbi_vi_quality_control=args.sbi_vi_quality_control,
        bayesflow_max_epochs=args.bayesflow_max_epochs,
        bayesflow_patience=args.bayesflow_patience,
        bayesflow_batch_size=args.bayesflow_batch_size,
        bayesflow_lr=args.bayesflow_lr,
        bayesflow_depth=args.bayesflow_depth,
        bayesflow_widths=args.bayesflow_widths or [],
        bayesflow_bins=args.bayesflow_bins,
        c2st_folds=args.c2st_folds,
        c2st_max_epochs=args.c2st_max_epochs,
        c2st_seed=args.c2st_seed,
        npe_pfn_filter_context_size=args.npe_pfn_filter_context_size,
        npe_pfn_n_estimators=args.npe_pfn_n_estimators,
    )


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    if not args.aggregate_only and not args.plot_only:
        total = len(args.tasks) * len(args.methods) * len(args.seeds) * len(args.budgets)
        idx = 0
        for task in args.tasks:
            for seed in args.seeds:
                for n_train in args.budgets:
                    for method in args.methods:
                        idx += 1
                        print(f"\n=== {idx}/{total} ===")
                        cfg = cfg_from_args(args, task, method, seed, n_train)
                        rows.append(run_cell(cfg, force=args.force))

    all_rows = load_all_cells()
    if rows:
        # Prefer freshly returned rows for cache state but include older cells too.
        keyed = {cell_variant_key(r): r for r in all_rows}
        for row in rows:
            keyed[cell_variant_key(row)] = row
        all_rows = list(keyed.values())

    if not all_rows:
        raise SystemExit("No cells available. Run without --aggregate-only/--plot-only first.")
    write_aggregate(all_rows, args.out_csv)
    plot_results(all_rows, args.fig_dir, tuple(args.formats))

    ok = [row for row in all_rows if row.get("status") == "ok"]
    print("\nSuccessful cells:")
    for row in sorted(ok, key=lambda r: (str(r["task"]), int(r["n_train"]), str(r["method"]), int(r["seed"]))):
        print(
            f"  {row['task']:<12} n={int(row['n_train']):<6} "
            f"{row['method']:<14} seed={int(row['seed']):<4} "
            f"joint={float(row['joint_c2st']):.3f} "
            f"total{int(row['n_ref'])}={float(row['total_seconds_n_ref']):.1f}s"
        )


if __name__ == "__main__":
    main()
