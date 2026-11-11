"""End-to-end ANN vs PINN experiment, configurable through `SHOConfig`.

The same config covers the simple (zeta = F0 = 0), damped (zeta > 0) and driven (F0 != 0)
harmonic oscillator, see `Oscillator`."""
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import torch

from .data import create_synthetic_data, create_NN_features, sho_initial_conditions
from .physics import Oscillator
from .models import build_model, train_ANN, train_PINN
from .evaluation import predict, ode_residual, metrics_table


@dataclass
class SHOConfig:
    """All the knobs of one ANN vs PINN experiment."""
    # ---- Oscillator parameters: x'' + 2 zeta omega x' + omega^2 x = F0 cos(Omega t) ----
    A: float = 32.0              # initial amplitude
    omega: float = 6.0           # natural angular frequency (rad/s)
    phi: float = 0.0             # initial phase: x(0) = A cos(phi), x'(0) = -A omega sin(phi)
    noise_std: float = None      # std dev of Gaussian noise (defaults to 0.3*A)
    zeta: float = 0.0            # damping ratio (0 <= zeta < 1)
    F0: float = 0.0              # driving amplitude (force per unit mass)
    Omega: float = 0.0           # driving angular frequency (rad/s)
    x0: float = None             # explicit initial conditions, override A and phi when given
    v0: float = None

    # ---- Training window (observed data), in periods T = 2*pi/omega ----
    n_periods_train: float = 1.0
    n_points: int = 400
    sampling_train: str = "uniform"
    sampling_train_kwargs: dict = field(default_factory=dict)
    seed_train: int = 42

    # ---- Test window (forecasting beyond the training data) ----
    period_factor: float = 0.5   # length of the test window as a fraction of the period
    n_test_points: int = 100
    sampling_test: str = "uniform"
    sampling_test_kwargs: dict = field(default_factory=dict)
    seed_test: int = 123         # different seed -> independent noise from the training set

    # ---- Shared model settings (identical for ANN and PINN) ----
    hidden_sizes: tuple = (16, 32, 16, 8)
    model_seed: int = 0          # same seed -> both models start from the same initial weights
    normalize: bool = True

    # ---- ANN training ----
    ann_lr: float = 1e-3
    ann_epochs: int = 1500

    # ---- PINN training ----
    pinn_lr: float = 5e-5
    pinn_epochs: int = 30_000
    lambda_physics: float = 1e-3
    lambda_ic: float = 1e-3
    n_colloc: int = 1000
    colloc_t_end: float = None   # end of collocation range (defaults to end of training window)
    colloc_test_window: bool = False  # True -> collocation up to the end of the test window
                                      # (only the ODE is used there, never the test data)

    def __post_init__(self):
        if self.noise_std is None:
            self.noise_std = 0.3 * self.A

    @property
    def T(self):
        """Natural period 2 pi / omega, the time unit of the training and test windows."""
        return 2 * np.pi / self.omega

    @property
    def oscillator(self):
        return Oscillator(self.omega, self.zeta, self.F0, self.Omega)

    @property
    def initial_conditions(self):
        """(x(0), x'(0)): x0 and v0 if given, otherwise from A and phi."""
        x0, v0 = sho_initial_conditions(self.A, self.omega, self.phi)
        return (x0 if self.x0 is None else self.x0), (v0 if self.v0 is None else self.v0)

    def with_sampling(self, sampling, **sampling_kwargs):
        """Copy of this config with a different training sampling scheme."""
        return replace(self, sampling_train=sampling, sampling_train_kwargs=sampling_kwargs)


@dataclass
class SHODataset:
    t: np.ndarray
    x_clean: np.ndarray
    x_noisy: np.ndarray
    t_test: np.ndarray
    x_test_clean: np.ndarray
    x_test_noisy: np.ndarray
    t_full: np.ndarray          # dense grid over training + test windows, used for plots
    x_full_true: np.ndarray
    t_start: float
    t_end: float
    t_test_start: float
    t_test_end: float


@dataclass
class ExperimentResult:
    config: SHOConfig
    data: SHODataset
    norm_stats: tuple           # (t_mean, t_std, x_mean, x_std)
    model_ann: torch.nn.Module
    model_pinn: torch.nn.Module
    losses_ann: list
    losses_pinn: dict
    predictions: dict           # keys: {ann,pinn}_{train,test,full}
    residual_ann: np.ndarray    # ODE residual over data.t_full
    residual_pinn: np.ndarray
    metrics: pd.DataFrame       # rows: sets, columns: (model, metric)
    physics_rms: pd.Series

    @property
    def noise_floor(self):
        """Data MSE (normalized) of the true signal: (sigma_noise / sigma_x)^2."""
        return (self.config.noise_std / float(self.norm_stats[3]))**2


@dataclass
class PINNRun:
    """One PINN trained with a given config (see `sweep_pinn`)."""
    config: SHOConfig
    data: SHODataset
    norm_stats: tuple           # (t_mean, t_std, x_mean, x_std)
    model: torch.nn.Module
    losses: dict                # 'data', 'physics', 'ic', 'total' per epoch (physics, IC unweighted)
    predictions: dict           # keys: train, test, full
    residual: np.ndarray        # ODE residual over data.t_full
    metrics: pd.DataFrame       # rows: sets, columns: metric

    @property
    def noise_floor(self):
        """Data MSE (normalized) of the true signal: (sigma_noise / sigma_x)^2."""
        return (self.config.noise_std / float(self.norm_stats[3]))**2


def generate_data(cfg, plot=False):
    """Training/test data and dense evaluation grid for a config."""
    t_start, t_end = 0.0, cfg.n_periods_train * cfg.T
    t_test_start, t_test_end = t_end, t_end + cfg.T * cfg.period_factor
    osc = cfg.oscillator
    x_fn = lambda t: osc.solution(t, *cfg.initial_conditions)

    t, x_clean, x_noisy = create_synthetic_data(
        x_fn, cfg.noise_std, t_start, t_end, cfg.n_points, seed=cfg.seed_train,
        sampling=cfg.sampling_train, sampling_kwargs=cfg.sampling_train_kwargs,
        plot_noise_dist=plot, plot_signal=plot, name=osc.name,
    )
    t_test, x_test_clean, x_test_noisy = create_synthetic_data(
        x_fn, cfg.noise_std, t_test_start, t_test_end, cfg.n_test_points,
        seed=cfg.seed_test, sampling=cfg.sampling_test, sampling_kwargs=cfg.sampling_test_kwargs,
        plot_noise_dist=False, plot_signal=plot, name=osc.name,
    )
    t_full = np.linspace(t_start, t_test_end, 1000)

    return SHODataset(t, x_clean, x_noisy, t_test, x_test_clean, x_test_noisy,
                      t_full, x_fn(t_full), t_start, t_end, t_test_start, t_test_end)


def run_experiment(cfg, plot_data=False, verbose=True):
    """
    Generate data, train the ANN and the PINN from the same initial weights and evaluate both.

    verbose : bool  Print training progress.
    """
    data = generate_data(cfg, plot=plot_data)
    t_norm, x_norm, t_mean, t_std, x_mean, x_std = create_NN_features(data.t, data.x_noisy, cfg.normalize)
    stats = (t_mean, t_std, x_mean, x_std)

    # ---- Model 1: plain ANN ----
    if verbose:
        print(f"[{cfg.sampling_train}] Training ANN ...")
    model_ann = build_model(cfg.hidden_sizes, cfg.model_seed)
    losses_ann = train_ANN(model_ann, t_norm, x_norm, lr=cfg.ann_lr, n_epochs=cfg.ann_epochs,
                           print_every=max(cfg.ann_epochs // 5, 1) if verbose else 0)

    # ---- Model 2: PINN ----
    if verbose:
        print(f"[{cfg.sampling_train}] Training PINN ...")
    model_pinn, losses_pinn = _train_pinn(cfg, data, t_norm, x_norm, stats, verbose)
    osc = cfg.oscillator

    # ---- Evaluation ----
    predictions = {}
    for name, model in (("ann", model_ann), ("pinn", model_pinn)):
        for split, t_eval in (("train", data.t), ("test", data.t_test), ("full", data.t_full)):
            predictions[f"{name}_{split}"] = predict(model, t_eval, *stats)

    residual_ann = ode_residual(model_ann, data.t_full, osc, *stats)
    residual_pinn = ode_residual(model_pinn, data.t_full, osc, *stats)

    metrics = pd.concat({
        "ANN":  metrics_table(data, predictions["ann_train"], predictions["ann_test"]),
        "PINN": metrics_table(data, predictions["pinn_train"], predictions["pinn_test"]),
    }, axis=1)
    physics_rms = pd.Series({"ANN": np.sqrt(np.mean(residual_ann**2)),
                             "PINN": np.sqrt(np.mean(residual_pinn**2))}, name="ODE residual RMS")

    return ExperimentResult(cfg, data, stats, model_ann, model_pinn, losses_ann, losses_pinn,
                            predictions, residual_ann, residual_pinn, metrics, physics_rms)


def _train_pinn(cfg, data, t_norm, x_norm, stats, verbose):
    """Build and train the PINN of a config. Returns (model, losses)."""
    t_mean, t_std, x_mean, x_std = stats
    # Collocation points: these can extend BEYOND the data range (set cfg.colloc_t_end)
    if cfg.colloc_test_window:
        colloc_end = data.t_test_end
    else:
        colloc_end = data.t_end if cfg.colloc_t_end is None else cfg.colloc_t_end
    t_colloc = torch.linspace(data.t_start, colloc_end, cfg.n_colloc).reshape(-1, 1)
    t_colloc_norm = ((t_colloc - t_mean) / t_std).requires_grad_(True)  # need grad for autodiff

    x0, v0 = cfg.initial_conditions
    x0, v0 = torch.tensor(x0, dtype=torch.float32), torch.tensor(v0, dtype=torch.float32)

    model = build_model(cfg.hidden_sizes, cfg.model_seed)
    losses = train_PINN(model, t_norm, x_norm, t_colloc_norm,
                        cfg.oscillator, t_mean, t_std, x_mean, x_std, x0, v0,
                        lambda_physics=cfg.lambda_physics, lambda_ic=cfg.lambda_ic,
                        lr=cfg.pinn_lr, n_epochs=cfg.pinn_epochs,
                        print_every=max(cfg.pinn_epochs // 10, 1) if verbose else 0)
    return model, losses


def run_pinn(cfg, verbose=False):
    """Train and evaluate only the PINN of a config (lambda_physics = lambda_ic = 0 is a plain ANN
    trained with the PINN optimizer settings)."""
    data = generate_data(cfg)
    t_norm, x_norm, *stats = create_NN_features(data.t, data.x_noisy, cfg.normalize)
    stats = tuple(stats)
    model, losses = _train_pinn(cfg, data, t_norm, x_norm, stats, verbose)

    predictions = {split: predict(model, t_eval, *stats)
                   for split, t_eval in (("train", data.t), ("test", data.t_test), ("full", data.t_full))}
    residual = ode_residual(model, data.t_full, cfg.oscillator, *stats)
    metrics = metrics_table(data, predictions["train"], predictions["test"])
    return PINNRun(cfg, data, stats, model, losses, predictions, residual, metrics)


def sweep_pinn(base_cfg, variants, verbose=False):
    """
    Train one PINN per config variant, everything else fixed (same data, same initial weights).

    variants : dict  name -> dict of `SHOConfig` overrides, e.g.
                     {"lambda_phys = 0": {"lambda_physics": 0.0}, "lambda_phys = 1e-3": {"lambda_physics": 1e-3}}

    Returns
    -------
    runs : dict  name -> PINNRun
    summary : pd.DataFrame  rows: variant, columns: loss weights and metrics
    """
    runs = {}
    for name, overrides in variants.items():
        print(f"Running {name} ...")
        runs[name] = run_pinn(replace(base_cfg, **overrides), verbose=verbose)
    return runs, summarize_runs(runs)


def summarize_runs(runs):
    """Loss weights, final losses and key metrics of several `PINNRun`s in one table."""
    rows = {}
    for name, run in runs.items():
        m, d, lp = run.metrics, run.data, run.losses
        test = d.t_full >= d.t_test_start
        rows[name] = {
            "lambda_physics": run.config.lambda_physics,
            "lambda_ic": run.config.lambda_ic,
            "Final data loss": lp["data"][-1],
            "Final physics loss": lp["physics"][-1],
            "Final IC loss": lp["ic"][-1],
            "Train RMSE (vs noisy)": m.loc["Train (vs noisy)", "RMSE"],
            "Train RMSE (vs true)": m.loc["Train (vs true)", "RMSE"],
            "Test RMSE (vs true)": m.loc["Test (vs true)", "RMSE"],
            "Test R2 (vs true)": m.loc["Test (vs true)", "R2"],
            "ODE residual RMS (train window)": np.sqrt(np.mean(run.residual[~test]**2)),
            "ODE residual RMS (test window)": np.sqrt(np.mean(run.residual[test]**2)),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def compare_sampling_schemes(base_cfg, schemes, verbose=False):
    """
    Run `run_experiment` once per training sampling scheme, everything else fixed.

    schemes : dict  name -> (sampling, sampling_kwargs), e.g.
                    {"beta(0.5,0.5)": ("beta", {"a": 0.5, "b": 0.5}), "uniform": ("uniform", {})}

    Returns
    -------
    results : dict  name -> ExperimentResult
    summary : pd.DataFrame  rows: scheme, columns: (metric, model)
    """
    results = {}
    for name, (sampling, kw) in schemes.items():
        print(f"Running {name} ...")
        results[name] = run_experiment(base_cfg.with_sampling(sampling, **(kw or {})), verbose=verbose)
    return results, summarize(results)


def summarize(results):
    """Key metrics (vs the true signal) of several experiments in one table."""
    rows = {}
    for name, res in results.items():
        m = res.metrics
        row = {}
        for model in ("ANN", "PINN"):
            row[("Train RMSE (vs true)", model)] = m.loc["Train (vs true)", (model, "RMSE")]
            row[("Test RMSE (vs true)", model)] = m.loc["Test (vs true)", (model, "RMSE")]
            row[("Test R2 (vs true)", model)] = m.loc["Test (vs true)", (model, "R2")]
            row[("ODE residual RMS", model)] = res.physics_rms[model]
        rows[name] = row
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.columns = pd.MultiIndex.from_tuples(summary.columns)
    return summary
