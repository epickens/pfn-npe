"""Shared SBIBM utilities for simulation, evaluation, and metrics.

Reusable by any inference method (TabPFN-NPE, NPE, BayesFlow, etc.).
Each method just needs to provide a `sample_fn(x_obs) -> samples` callable
to use `evaluate_posterior`.

ODE tasks (sir, lotka_volterra) use Python/scipy simulators instead of
Julia/diffeqtorch, matching sbibm's exact specifications.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import sbibm
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.stats import kstest
from sklearn.model_selection import KFold

from pfn_testing.sbi.custom_tasks.distractor_task import DistractorTask


# ═══════════════════════════════════════════════════════════════════════════════
# Task registry
# ═══════════════════════════════════════════════════════════════════════════════

AVAILABLE_TASKS: list[str] = [
    "two_moons",
    "gaussian_mixture",
    "slcp",
    "slcp_distractors",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "bernoulli_glm",
    "bernoulli_glm_raw",
    "sir",
    "lotka_volterra",
    "ar1_ts_t50",
    "ou",
    "solar_dynamo",
    "sir_raw",
    "lotka_volterra_raw",
    "two_moons_distractors",
    "gaussian_mixture_distractors",
    "bernoulli_glm_distractors",
    "sir_distractors",
]

# Tasks whose simulator we replace with Python (Julia/diffeqtorch is broken)
ODE_TASKS: set[str] = {"sir", "lotka_volterra"}

# Lazy-initialized custom tasks (avoid expensive init at import time)
_CUSTOM_TASK_FACTORIES: dict[str, Callable] = {
    "ar1_ts_t50": lambda: __import__(
        "pfn_testing.sbi.custom_tasks.ar1_ts_t50", fromlist=["AR1TimeSeriesT50Task"]
    ).AR1TimeSeriesT50Task(seed=12345, n_obs=10, n_ref=10_000),
    "ou": lambda: __import__(
        "pfn_testing.sbi.custom_tasks.ou", fromlist=["OUTask"]
    ).OUTask(seed=12345, n_obs=10, n_ref=10_000),
    "solar_dynamo": lambda: __import__(
        "pfn_testing.sbi.custom_tasks.solar_dynamo", fromlist=["SolarDynamoTask"]
    ).SolarDynamoTask(seed=12345, n_obs=10, n_ref=10_000),
    "sir_raw": lambda: __import__(
        "pfn_testing.sbi.custom_tasks.sir_raw", fromlist=["SIRRawTask"]
    ).SIRRawTask(seed=12345, n_obs=10, n_ref=10_000),
    "lotka_volterra_raw": lambda: __import__(
        "pfn_testing.sbi.custom_tasks.lotka_volterra_raw", fromlist=["LotkaVolterraRawTask"]
    ).LotkaVolterraRawTask(seed=12345, n_obs=10, n_ref=10_000),
    "two_moons_distractors": lambda: DistractorTask("two_moons", noise_dim=92),
    "gaussian_mixture_distractors": lambda: DistractorTask("gaussian_mixture", noise_dim=92),
    "bernoulli_glm_distractors": lambda: DistractorTask("bernoulli_glm", noise_dim=90),
    "sir_distractors": lambda: DistractorTask("sir", noise_dim=90),
}
_CUSTOM_TASK_CACHE: dict[str, object] = {}

def get_task(task_name: str):
    """Get task object, checking custom tasks first, then sbibm."""
    if task_name in _CUSTOM_TASK_CACHE:
        return _CUSTOM_TASK_CACHE[task_name]
    if task_name in _CUSTOM_TASK_FACTORIES:
        task = _CUSTOM_TASK_FACTORIES[task_name]()
        _CUSTOM_TASK_CACHE[task_name] = task
        return task
    _patch_torch_load_for_legacy_files()
    return sbibm.get_task(task_name)

# Tasks that need torch.load(weights_only=False) for legacy pyro GMM files
_TORCH_LOAD_PATCHED = False

def _patch_torch_load_for_legacy_files() -> None:
    """Patch torch.load to allow legacy sbibm files (gmm.torch, permutation_idx.torch).

    Needed for slcp_distractors which stores a pyro GMM distribution via torch.save,
    incompatible with PyTorch 2.6+ weights_only=True default.
    """
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return
    _original = torch.load

    def _patched(f, *args, **kwargs):
        if isinstance(f, (str, type(None))) or hasattr(f, "__fspath__"):
            path_str = str(f)
            if "gmm.torch" in path_str or "permutation_idx.torch" in path_str:
                kwargs["weights_only"] = False
        return _original(f, *args, **kwargs)

    torch.load = _patched  # type: ignore[assignment]
    _TORCH_LOAD_PATCHED = True


# ═══════════════════════════════════════════════════════════════════════════════
# Python ODE simulators (replacing Julia/diffeqtorch)
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_sir_single(
    beta: float, gamma: float, rng: np.random.Generator,
    subsample_step: int = 17,
) -> np.ndarray | None:
    """Simulate a single SIR trajectory matching sbibm specifications.

    ODE: dS/dt = -beta*S*I/N, dI/dt = beta*S*I/N - gamma*I, dR/dt = gamma*I
    Returns observation vector or None if solver fails.
    With subsample_step=17: (10,) matching sbibm. With subsample_step=1: (161,) raw.
    """
    N = 1_000_000
    u0 = [N - 1, 1, 0]  # [S0, I0, R0]
    tspan = (0.0, 160.0)
    t_eval = np.arange(0, 161, dtype=float)  # saveat=1.0, 161 points

    def ode(t, u):
        S, I, R = u
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return [dS, dI, dR]

    sol = solve_ivp(ode, tspan, u0, t_eval=t_eval, method="RK45",
                    rtol=1e-8, atol=1e-8)

    if sol.status != 0 or sol.y.shape != (3, 161):
        return None

    infected = sol.y[1, ::subsample_step]

    # Binomial observation noise: Binomial(n=1000, p=I/N)
    probs = np.clip(infected / N, 0, 1)
    obs = rng.binomial(n=1000, p=probs).astype(float)
    return obs


def _simulate_lotka_volterra_single(
    alpha: float, beta: float, gamma: float, delta: float,
    rng: np.random.Generator,
    subsample_step: int = 21,
) -> np.ndarray | None:
    """Simulate a single Lotka-Volterra trajectory matching sbibm specs.

    ODE: dx/dt = alpha*x - beta*x*y, dy/dt = -gamma*y + delta*x*y
    Returns observation vector or None if solver fails.
    With subsample_step=21: (20,) matching sbibm. With subsample_step=1: (402,) raw.
    """
    u0 = [30.0, 1.0]  # [prey, predator]
    tspan = (0.0, 20.0)
    t_eval = np.arange(0, 20.01, 0.1)  # saveat=0.1, 201 points

    def ode(t, u):
        x, y = u
        dx = alpha * x - beta * x * y
        dy = -gamma * y + delta * x * y
        return [dx, dy]

    sol = solve_ivp(ode, tspan, u0, t_eval=t_eval, method="RK45",
                    rtol=1e-8, atol=1e-8)

    if sol.status != 0 or sol.y.shape[1] < 201:
        return None

    prey = sol.y[0, ::subsample_step]
    predator = sol.y[1, ::subsample_step]
    raw = np.concatenate([prey, predator])

    # LogNormal observation noise: LogNormal(log(clamp(x, 1e-10, 10000)), 0.1)
    raw_clamped = np.clip(raw, 1e-10, 10000)
    obs = rng.lognormal(mean=np.log(raw_clamped), sigma=0.1)
    return obs


def _simulate_ode_task(
    task_name: str,
    thetas: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Run Python ODE simulator for a batch of parameters.

    Args:
        task_name: "sir" or "lotka_volterra"
        thetas: (n_samples, dim_theta) array of parameters
        seed: random seed for observation noise

    Returns:
        xs: (n_samples, dim_x) array of observations
    """
    rng = np.random.default_rng(seed)
    n = len(thetas)

    if task_name == "sir":
        dim_x = 10
        xs = np.full((n, dim_x), np.nan)
        for i in range(n):
            beta, gamma = thetas[i]
            result = _simulate_sir_single(beta, gamma, rng)
            if result is not None:
                xs[i] = result
    elif task_name == "lotka_volterra":
        dim_x = 20
        xs = np.full((n, dim_x), np.nan)
        for i in range(n):
            alpha, beta, gamma, delta = thetas[i]
            result = _simulate_lotka_volterra_single(alpha, beta, gamma, delta, rng)
            if result is not None:
                xs[i] = result
    else:
        raise ValueError(f"Unknown ODE task: {task_name}")

    # Remove failed simulations (NaN rows)
    valid_mask = ~np.isnan(xs).any(axis=1)
    n_failed = n - valid_mask.sum()
    if n_failed > 0:
        print(f"  Warning: {n_failed}/{n} simulations failed (NaN), keeping {valid_mask.sum()}")

    return xs, valid_mask


# ═══════════════════════════════════════════════════════════════════════════════
# Task info
# ═══════════════════════════════════════════════════════════════════════════════

# Pre-defined info for tasks that are expensive to instantiate
_TASK_INFO: dict[str, dict] = {
    "sir": {"dim_theta": 2, "dim_x": 10, "n_obs": 10},
    "lotka_volterra": {"dim_theta": 4, "dim_x": 20, "n_obs": 10},
    "sir_raw": {"dim_theta": 2, "dim_x": 161, "n_obs": 10},
    "lotka_volterra_raw": {"dim_theta": 4, "dim_x": 402, "n_obs": 10},
    "ou": {"dim_theta": 3, "dim_x": 100, "n_obs": 10},
    "solar_dynamo": {"dim_theta": 3, "dim_x": 100, "n_obs": 10},
}


def get_task_info(task_name: str) -> dict:
    """Get task dimensions without running simulations.

    Returns:
        {"task_name", "dim_theta", "dim_x", "n_obs"}
    """
    if task_name in _TASK_INFO:
        return {"task_name": task_name, **_TASK_INFO[task_name]}

    task = get_task(task_name)
    prior = task.get_prior()
    simulator = task.get_simulator()
    theta_sample = prior(num_samples=1)
    x_sample = simulator(theta_sample)
    return {
        "task_name": task_name,
        "dim_theta": theta_sample.shape[1],
        "dim_x": x_sample.shape[1],
        "n_obs": 10,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(
    task_name: str,
    n_train: int = 10_000,
    n_val: int = 2_000,
    seed: int = 42,
) -> dict:
    """Simulate training and validation data from an sbibm task.

    For ODE tasks (sir, lotka_volterra), uses Python/scipy simulators.
    For all others, uses sbibm's built-in simulators.

    Returns:
        {"thetas_train", "xs_train", "thetas_val", "xs_val", "task",
         "dim_theta", "dim_x"}
    """
    task = get_task(task_name)
    prior = task.get_prior()

    n_total = n_train + n_val
    torch.manual_seed(seed)

    # Check if this is an ODE-based task (directly or via distractor wrapper)
    is_ode = task_name in ODE_TASKS
    if not is_ode and hasattr(task, "base_task_name") and task.base_task_name in ODE_TASKS:
        is_ode = True

    if is_ode:
        # Over-sample to account for failed simulations
        n_oversample = int(n_total * 1.2)
        thetas_all = prior(num_samples=n_oversample).numpy()

        if task_name in ODE_TASKS:
            xs_all, valid_mask = _simulate_ode_task(task_name, thetas_all, seed)
        else:
            # ODE-based custom task (e.g. sir_distractors) — use its simulator
            simulator = task.get_simulator()
            xs_t = simulator(torch.tensor(thetas_all, dtype=torch.float32))
            xs_all = xs_t.numpy()
            valid_mask = ~np.isnan(xs_all).any(axis=1)

        thetas_valid = thetas_all[valid_mask]
        xs_valid = xs_all[valid_mask]

        if len(thetas_valid) < n_total:
            raise RuntimeError(
                f"Only {len(thetas_valid)}/{n_total} simulations succeeded. "
                "Try increasing n_train or check ODE parameters."
            )
        thetas = thetas_valid[:n_total]
        xs = xs_valid[:n_total]
    else:
        simulator = task.get_simulator()
        thetas = prior(num_samples=n_total).numpy()
        xs = simulator(torch.tensor(thetas, dtype=torch.float32)).numpy()

    return {
        "thetas_train": thetas[:n_train],
        "xs_train": xs[:n_train],
        "thetas_val": thetas[n_train:],
        "xs_val": xs[n_train:],
        "task": task,
        "dim_theta": thetas.shape[1],
        "dim_x": xs.shape[1],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# C2ST metric
# ═══════════════════════════════════════════════════════════════════════════════

def compute_c2st(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n_folds: int = 5,
    max_epochs: int = 500,
    lr: float = 1e-3,
    seed: int = 1,
) -> float:
    """Classifier Two-Sample Test. 0.5 = perfect, 1.0 = completely different.

    PyTorch MLP matching sbibm's spec: 2 hidden layers of 10*ndim ReLU units,
    z-scored inputs, 5-fold CV. Runs on GPU if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ndim = samples_a.shape[1]
    n = min(len(samples_a), len(samples_b))

    # Z-score using samples_a statistics (matches sbibm)
    mean = samples_a.mean(axis=0)
    std = samples_a.std(axis=0) + 1e-8
    a = (samples_a[:n] - mean) / std
    b = (samples_b[:n] - mean) / std

    X = np.concatenate([a, b], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []

    # enable_grad needed because BayesFlow disables autograd globally at import
    with torch.enable_grad():
        for train_idx, val_idx in kf.split(X):
            X_tr  = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
            y_tr  = torch.tensor(y[train_idx],  dtype=torch.float32, device=device)
            X_val = torch.tensor(X[val_idx],   dtype=torch.float32, device=device)
            y_val = torch.tensor(y[val_idx],   dtype=torch.float32, device=device)

            model = nn.Sequential(
                nn.Linear(ndim, 10 * ndim), nn.ReLU(),
                nn.Linear(10 * ndim, 10 * ndim), nn.ReLU(),
                nn.Linear(10 * ndim, 1),
            ).to(device)

            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.BCEWithLogitsLoss()

            best_loss = float("inf")
            no_improve = 0
            for _ in range(max_epochs):
                model.train()
                opt.zero_grad()
                loss = loss_fn(model(X_tr).squeeze(1), y_tr)
                loss.backward()
                opt.step()

                if loss.item() < best_loss - 1e-4:
                    best_loss = loss.item()
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= 30:
                    break

            model.eval()
            with torch.no_grad():
                preds = (model(X_val).squeeze(1) > 0).float()
                fold_accs.append((preds == y_val).float().mean().item())

    return float(np.mean(fold_accs))


# ═══════════════════════════════════════════════════════════════════════════════
# Posterior evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_posterior(
    task: object,
    sample_fn: Callable[[np.ndarray], np.ndarray],
    n_obs: int = 10,
    n_posterior_samples: int = 10_000,
    verbose: bool = True,
) -> list[float]:
    """Evaluate a posterior approximation against sbibm reference posteriors.

    Args:
        task: sbibm task object (from sbibm.get_task())
        sample_fn: callable that takes x_obs (1D array, shape (dim_x,)) and
            returns posterior samples (2D array, shape (n_samples, dim_theta))
        n_obs: number of sbibm reference observations to evaluate
        n_posterior_samples: how many samples to draw per observation
        verbose: print per-observation scores

    Returns:
        List of C2ST scores, one per observation.
    """
    c2st_scores = []

    for i in range(1, n_obs + 1):
        x_obs = task.get_observation(num_observation=i)
        ref_samples = task.get_reference_posterior_samples(num_observation=i)

        x_obs_np = x_obs.numpy().squeeze(0)
        ref_np = ref_samples.numpy()

        posterior_samples = sample_fn(x_obs_np)
        if len(posterior_samples) > n_posterior_samples:
            posterior_samples = posterior_samples[:n_posterior_samples]

        score = compute_c2st(posterior_samples, ref_np)
        c2st_scores.append(score)

        if verbose:
            print(f"    Obs {i}: C2ST={score:.4f}")

    return c2st_scores


# ═══════════════════════════════════════════════════════════════════════════════
# Single-sample simulation (for SBC)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_single(
    task_name: str,
    theta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Simulate a single observation from the given parameters.

    Args:
        task_name: SBIBM task name
        theta: 1D array of parameters, shape (dim_theta,)
        rng: numpy random generator for stochastic simulators

    Returns:
        1D array of observations, shape (dim_x,), or None if simulation fails
    """
    if task_name in _CUSTOM_TASK_FACTORIES:
        task = get_task(task_name)
        simulator = task.get_simulator()
        theta_t = torch.tensor(theta, dtype=torch.float32).unsqueeze(0)
        x = simulator(theta_t)
        return x.numpy().squeeze(0)

    if task_name == "sir":
        return _simulate_sir_single(theta[0], theta[1], rng)
    elif task_name == "lotka_volterra":
        return _simulate_lotka_volterra_single(
            theta[0], theta[1], theta[2], theta[3], rng,
        )
    else:
        _patch_torch_load_for_legacy_files()
        task = sbibm.get_task(task_name)
        simulator = task.get_simulator()
        theta_t = torch.tensor(theta, dtype=torch.float32).unsqueeze(0)
        x = simulator(theta_t)
        return x.numpy().squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation-Based Calibration (SBC)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sbc(
    task_name: str,
    sample_fn: Callable[[np.ndarray], np.ndarray],
    n_trials: int = 1000,
    n_posterior_samples: int = 1000,
    seed: int = 123,
    verbose: bool = True,
) -> dict:
    """Run Simulation-Based Calibration (Talts et al., 2018).

    For each trial: sample theta_true from prior, simulate x, draw posterior
    samples, compute the rank of theta_true among samples. If the posterior is
    well-calibrated, ranks should be uniformly distributed.

    Args:
        task_name: SBIBM task name
        sample_fn: callable taking x_obs (1D) -> posterior samples (2D)
        n_trials: number of SBC trials
        n_posterior_samples: posterior samples per trial
        seed: random seed for prior/simulator
        verbose: print progress

    Returns:
        dict with keys:
            ranks: (n_valid_trials, dim_theta) array of ranks
            ks_stats: (dim_theta,) KS statistic per dimension
            ks_pvalues: (dim_theta,) KS p-value per dimension
            mean_ks: float, mean KS stat across dimensions
            n_failed: int, number of failed simulations
    """
    task = get_task(task_name)
    prior = task.get_prior()
    rng = np.random.default_rng(seed)

    ranks_list: list[np.ndarray] = []
    n_failed = 0

    for i in range(n_trials):
        # Sample theta from prior
        torch.manual_seed(seed + i)
        theta_true = prior(num_samples=1).numpy().squeeze(0)

        # Simulate observation
        x_obs = simulate_single(task_name, theta_true, rng)
        if x_obs is None:
            n_failed += 1
            continue

        # Draw posterior samples
        posterior_samples = sample_fn(x_obs)
        if len(posterior_samples) > n_posterior_samples:
            posterior_samples = posterior_samples[:n_posterior_samples]

        # Compute rank: count how many posterior samples are less than theta_true
        rank = (posterior_samples < theta_true).sum(axis=0)
        ranks_list.append(rank)

        if verbose and (i + 1) % 100 == 0:
            print(f"    SBC trial {i + 1}/{n_trials}")

    ranks = np.array(ranks_list)  # (n_valid, dim_theta)
    dim_theta = ranks.shape[1]

    # KS test against uniform for each dimension
    ks_stats = np.zeros(dim_theta)
    ks_pvalues = np.zeros(dim_theta)
    for d in range(dim_theta):
        # Normalize ranks to [0, 1] for KS test
        normalized = ranks[:, d] / n_posterior_samples
        stat, pval = kstest(normalized, "uniform")
        ks_stats[d] = stat
        ks_pvalues[d] = pval

    mean_ks = float(ks_stats.mean())

    if verbose:
        print(f"    SBC complete: {len(ranks)}/{n_trials} valid trials, "
              f"mean KS={mean_ks:.4f}")
        if n_failed > 0:
            print(f"    ({n_failed} simulations failed)")

    return {
        "ranks": ranks,
        "ks_stats": ks_stats,
        "ks_pvalues": ks_pvalues,
        "mean_ks": mean_ks,
        "n_failed": n_failed,
    }
