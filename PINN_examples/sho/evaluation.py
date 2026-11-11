"""Inference, metrics and plots. Plot functions named `plot_*` for a single experiment take an
`ExperimentResult` (see `experiment.py`); the rest take plain arrays."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from .models import physics_residual

ANN_COLOR, PINN_COLOR, TRUE_COLOR = "blue", "red", "green"


# ---------------------------------------------------------------------------
# Inference & metrics
# ---------------------------------------------------------------------------
def predict(model, t, t_mean, t_std, x_mean, x_std):
    """
    Run inference on raw times t (np.ndarray) and return predictions in physical units.
    Inputs are normalized with the TRAINING statistics and outputs are unnormalized.
    """
    model.eval()
    t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    t_norm = (t_tensor - t_mean) / t_std
    with torch.no_grad():
        x_pred_norm = model(t_norm)
    return (x_pred_norm * x_std + x_mean).numpy().flatten()


def predict_velocity(model, t, t_mean, t_std, x_mean, x_std):
    """dx/dt of a trained model on raw times t (np.ndarray), in physical units (via autograd)."""
    model.eval()
    t_norm = ((torch.tensor(t, dtype=torch.float32).reshape(-1, 1) - t_mean) / t_std).requires_grad_(True)
    x_norm = model(t_norm)
    dx_norm = torch.autograd.grad(x_norm, t_norm, grad_outputs=torch.ones_like(x_norm))[0]
    return ((x_std / t_std) * dx_norm).detach().numpy().flatten()


def ode_residual(model, t, osc, t_mean, t_std, x_mean, x_std):
    """ODE residual of `osc` (an `Oscillator`) for a trained model on raw times t (np.ndarray)."""
    t_norm = ((torch.tensor(t, dtype=torch.float32).reshape(-1, 1) - t_mean) / t_std).requires_grad_(True)
    return physics_residual(model, t_norm, osc, t_mean, t_std, x_mean, x_std).detach().numpy().flatten()


def compute_metrics(x_true, x_pred):
    """Regression metrics in physical units."""
    err = x_pred - x_true
    mse = np.mean(err**2)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": np.mean(np.abs(err)),
        "R2": 1 - np.sum(err**2) / np.sum((x_true - x_true.mean())**2),
    }


def metrics_table(data, x_train_pred, x_test_pred):
    """Metrics of one model on train/test sets, against noisy data and the true signal."""
    return pd.DataFrame({
        "Train (vs noisy)": compute_metrics(data.x_noisy, x_train_pred),
        "Train (vs true)":  compute_metrics(data.x_clean, x_train_pred),
        "Test (vs noisy)":  compute_metrics(data.x_test_noisy, x_test_pred),
        "Test (vs true)":   compute_metrics(data.x_test_clean, x_test_pred),
    }).T


# ---------------------------------------------------------------------------
# Generic plots (arrays)
# ---------------------------------------------------------------------------
def plot_noise_distribution(noise):
    plt.figure(figsize=(7, 4))
    sns.histplot(noise, kde=True, stat="density", bins=15, color="steelblue", edgecolor="white")
    plt.xlabel("Noise value")
    plt.ylabel("Density")
    plt.title("Distribution of Gaussian Noise")
    plt.grid(alpha=0.8)
    plt.show()


def plot_signal(t, x_clean, x_noisy, title="Synthetic Simple Harmonic Oscillator Data"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(t, x_clean, label="Clean signal", color="steelblue", s=15, alpha=1)
    ax.scatter(t, x_noisy, color="orange", s=15, label="Noisy data", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.8)
    plt.show()


def plot_fit(t, x_noisy, x_clean, x_pred, title, data_label="Noisy data", pred_label="NN prediction",
             true_label="True signal"):
    """Prediction vs data/ground truth (top) and residuals w.r.t. ground truth (bottom)."""
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax.scatter(t, x_noisy, s=15, color="orange", alpha=0.5, label=data_label)
    ax.plot(t, x_clean, color=TRUE_COLOR, linewidth=2, linestyle="--", label=true_label)
    ax.plot(t, x_pred, color="blue", linewidth=2, alpha=0.7, label=pred_label)
    ax.set_ylabel("Displacement")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(alpha=0.8)

    ax_res.plot(t, x_pred - x_clean, color="red", linewidth=1.5)
    ax_res.axhline(0, color="gray", linestyle=":")
    ax_res.set_xlabel("Time (s)")
    ax_res.set_ylabel("Pred - True")
    ax_res.grid(alpha=0.8)

    plt.tight_layout()
    plt.show()


def plot_sampling_schemes(schemes, t_start, t_end, n_points, seed=42, A=1.0, omega=None, x_fn=None):
    """
    Visualize where each sampling scheme places the time samples.

    schemes : dict      name -> (sampling, sampling_kwargs)
    omega   : float     If given, the signal A cos(omega t) is drawn behind the samples for reference.
    x_fn    : callable  Any other reference signal t -> x (takes precedence over omega).
    """
    from .data import sample_time_points

    n = len(schemes)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.4 * n), sharex=True, squeeze=False)
    t_dense = np.linspace(t_start, t_end, 500)
    for ax, (name, (sampling, kw)) in zip(axes[:, 0], schemes.items()):
        t = sample_time_points(t_start, t_end, n_points, sampling,
                               rng=np.random.default_rng([seed, 1]), **(kw or {}))
        if x_fn is None and omega is not None:
            x_fn = lambda tt: A * np.cos(omega * tt)
        if x_fn is not None:
            ax.plot(t_dense, x_fn(t_dense), color="lightgray", zorder=0)
            ax.scatter(t, x_fn(t), s=6, color="steelblue")
        else:
            ax.eventplot(t, lineoffsets=0, linelengths=1, linewidths=0.6, color="steelblue")
            ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(f"Training time sampling schemes (n = {n_points})")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Single-experiment plots (ExperimentResult)
# ---------------------------------------------------------------------------
def _mark_test_window(ax, res, label=True):
    d = res.data
    ax.axvspan(d.t_test_start, d.t_test_end, color="gray", alpha=0.1,
               label="Test window" if label else None)
    ax.axvline(d.t_end, color="gray", linestyle=":", alpha=0.7,
               label="End of training range" if label else None)


def plot_ann_loss(res):
    plt.figure(figsize=(6, 4))
    plt.plot(res.losses_ann)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("ANN - Training Loss")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.show()


def plot_pinn_losses(res):
    lp, cfg = res.losses_pinn, res.config
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(lp["data"], label="Data loss")
    ax.plot(np.array(lp["physics"]) * cfg.lambda_physics, label="Physics loss (weighted)")
    ax.plot(np.array(lp["ic"]) * cfg.lambda_ic, label="IC loss (weighted)")
    ax.plot(lp["total"], label="Total loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PINN - Training Loss Components")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_fit(res, model="ann", split="train"):
    """plot_fit for one model ('ann' | 'pinn') on one split ('train' | 'test')."""
    d, name = res.data, model.upper()
    pred = res.predictions[f"{model}_{split}"]
    if split == "train":
        plot_fit(d.t, d.x_noisy, d.x_clean, pred, title=f"{name} Fit - Training Set",
                 data_label="Noisy training data", pred_label=f"{name} prediction",
                 true_label=f"True {res.config.oscillator.name}")
    else:
        plot_fit(d.t_test, d.x_test_noisy, d.x_test_clean, pred,
                 title=f"{name} Predictions - Test Set (out-of-sample)",
                 data_label="Noisy test data", pred_label=f"{name} prediction",
                 true_label=f"True {res.config.oscillator.name}")


def plot_data_loss_vs_noise_floor(res):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(res.losses_ann, color=ANN_COLOR, label="ANN - data loss")
    ax.plot(res.losses_pinn["data"], color=PINN_COLOR, label="PINN - data loss")
    ax.axhline(res.noise_floor, color="black", linestyle=":",
               label="Noise floor $(\\sigma_{noise}/\\sigma_x)^2$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Data MSE (normalized)")
    ax.set_title("Data Loss: ANN vs PINN")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions_full(res, ax=None, legend=True, title=None):
    """Both models' predictions over the training + test windows."""
    d, p = res.data, res.predictions
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(16, 6))
    _mark_test_window(ax, res)
    ax.scatter(d.t, d.x_noisy, s=10, color="orange", alpha=0.4, label="Training data")
    ax.scatter(d.t_test, d.x_test_noisy, s=10, color="purple", alpha=0.4, label="Test data")
    ax.plot(d.t_full, d.x_full_true, color=TRUE_COLOR, linewidth=2, linestyle="--", label="True signal")
    ax.plot(d.t_full, p["ann_full"], color=ANN_COLOR, linewidth=2, alpha=0.6, label="ANN prediction")
    ax.plot(d.t_full, p["pinn_full"], color=PINN_COLOR, linewidth=2, alpha=0.7, label="PINN prediction")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement")
    ax.set_title(title or f"ANN vs PINN - Training vs Test Range ({res.config.sampling_train} sampling)")
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(alpha=0.8)
    if standalone:
        plt.tight_layout()
        plt.show()


def plot_error_full(res):
    d, p = res.data, res.predictions
    fig, ax = plt.subplots(figsize=(16, 4))
    _mark_test_window(ax, res)
    ax.plot(d.t_full, p["ann_full"] - d.x_full_true, color=ANN_COLOR, linewidth=1.5, label="ANN")
    ax.plot(d.t_full, p["pinn_full"] - d.x_full_true, color=PINN_COLOR, linewidth=1.5, label="PINN")
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pred - True")
    ax.set_title("Prediction Error: ANN vs PINN")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(alpha=0.8)
    plt.tight_layout()
    plt.show()


def plot_ode_residual(res):
    d = res.data
    fig, ax = plt.subplots(figsize=(16, 4))
    _mark_test_window(ax, res)
    ax.plot(d.t_full, np.abs(res.residual_ann), color=ANN_COLOR, linewidth=1.5, label="ANN")
    ax.plot(d.t_full, np.abs(res.residual_pinn), color=PINN_COLOR, linewidth=1.5, label="PINN")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(res.config.oscillator.residual_label)
    ax.set_title("ODE Residual: ANN vs PINN")
    ax.set_yscale("log")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_phase_portrait(res):
    """Phase portrait (x, dx/dt) over training + test windows: true signal, ANN and PINN."""
    d, cfg = res.data, res.config
    v_true = cfg.oscillator.velocity(d.t_full, *cfg.initial_conditions)
    curves = [("True signal", d.x_full_true, v_true, TRUE_COLOR),
              ("ANN", res.predictions["ann_full"], predict_velocity(res.model_ann, d.t_full, *res.norm_stats), ANN_COLOR),
              ("PINN", res.predictions["pinn_full"], predict_velocity(res.model_pinn, d.t_full, *res.norm_stats), PINN_COLOR)]
    test = d.t_full >= d.t_test_start

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, (name, x, v, color) in zip(axes, curves):
        ax.plot(x[~test], v[~test], color=color, linewidth=1.2, label="Training window")
        ax.plot(x[test], v[test], color=color, linewidth=1.2, linestyle="--", alpha=0.6, label="Test window")
        ax.plot(x[0], v[0], "o", color="black", label="t = 0")
        ax.set_title(name)
        ax.set_xlabel("Displacement $x$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Velocity $\\dot{x}$")
    axes[0].legend(loc="upper right")
    fig.suptitle("Phase portrait")
    plt.tight_layout()
    plt.show()


def plot_power_spectrum(res):
    """
    Power spectrum |FFT(x)|^2 over training + test windows, against angular frequency.
    Dotted lines mark the transient (omega_d) and driving (Omega) frequencies.
    """
    d, osc = res.data, res.config.oscillator
    dt = d.t_full[1] - d.t_full[0]
    w = 2 * np.pi * np.fft.rfftfreq(d.t_full.size, dt)

    fig, ax = plt.subplots(figsize=(16, 4))
    for name, x, color, style in (("True signal", d.x_full_true, TRUE_COLOR, "--"),
                                  ("ANN", res.predictions["ann_full"], ANN_COLOR, "-"),
                                  ("PINN", res.predictions["pinn_full"], PINN_COLOR, "-")):
        power = np.abs(np.fft.rfft(x - x.mean()))**2
        ax.plot(w, power / power.max(), color=color, linestyle=style, linewidth=1.5, label=name)
    ax.axvline(osc.omega_d, color="gray", linestyle=":", label="$\\omega_d$ (transient)")
    if osc.F0:
        ax.axvline(osc.Omega, color="black", linestyle=":", label="$\\Omega$ (driving)")
    ax.set_xlim(0, 3 * max(osc.omega, osc.Omega))
    ax.set_xlabel("Angular frequency (rad/s)")
    ax.set_ylabel("Power (normalized)")
    ax.set_title("Power spectrum")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Multi-experiment plots (dict name -> ExperimentResult)
# ---------------------------------------------------------------------------
def plot_predictions_grid(results, ncols=2):
    """One `plot_predictions_full` panel per experiment."""
    n = len(results)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 3.5 * nrows),
                             squeeze=False, sharey=True)
    for ax, (name, res) in zip(axes.flat, results.items()):
        plot_predictions_full(res, ax=ax, legend=False, title=name)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.show()


def plot_summary_bars(summary, metric="Test RMSE (vs true)"):
    """Bar chart ANN vs PINN of one metric of `compare_sampling_schemes`'s summary table."""
    df = summary[metric].sort_values("PINN")
    ax = df.plot.bar(figsize=(12, 4), color=[ANN_COLOR, PINN_COLOR], alpha=0.7, rot=30)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by training sampling scheme")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Loss-weight sweep plots (dict name -> PINNRun, summary of `sweep_pinn`)
# ---------------------------------------------------------------------------
def _sweep_colors(runs):
    return dict(zip(runs, plt.cm.viridis(np.linspace(0, 0.9, len(runs)))))


def plot_lambda_sweep(summary, noise_std=None, noise_floor=None):
    """
    Metrics and final losses against lambda_physics (log axis). Rows with lambda_physics = 0
    can't sit on a log axis, so they are drawn as horizontal lines labelled with their name.
    """
    on_axis = summary[summary["lambda_physics"] > 0].sort_values("lambda_physics")
    baselines = summary[summary["lambda_physics"] == 0]
    panels = [
        ("Fit on the training window", ["Train RMSE (vs noisy)", "Train RMSE (vs true)"], "RMSE",
         ("Noise std", noise_std)),
        ("Forecast on the test window", ["Test RMSE (vs true)"], "RMSE", None),
        ("Physical consistency", ["ODE residual RMS (train window)", "ODE residual RMS (test window)"],
         "ODE residual RMS", None),
        ("Final training losses", ["Final data loss", "Final physics loss"], "Loss (physics unweighted)",
         ("Noise floor", noise_floor)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for ax, (title, cols, ylabel, ref) in zip(axes.flat, panels):
        for col, marker in zip(cols, ("o", "s")):
            line, = ax.plot(on_axis["lambda_physics"], on_axis[col], marker=marker, label=col)
            for (name, row), style in zip(baselines.iterrows(), ("--", ":", "-.")):
                ax.axhline(row[col], color=line.get_color(), linestyle=style, alpha=0.6,
                           label=f"{col} — {name}")
        if ref and ref[1] is not None:
            ax.axhline(ref[1], color="black", linestyle=(0, (1, 3)), label=ref[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    for ax in axes[-1]:
        ax.set_xlabel("$\\lambda_{phys}$")
    plt.tight_layout()
    plt.show()


def plot_sweep_predictions(runs, ncols=2):
    """One panel per run: data, true signal and prediction over the training + test windows."""
    n = len(runs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 3.2 * nrows), squeeze=False,
                             sharey=True, sharex=True)
    colors = _sweep_colors(runs)
    for ax, (name, run) in zip(axes.flat, runs.items()):
        d = run.data
        _mark_test_window(ax, run)
        ax.scatter(d.t, d.x_noisy, s=8, color="orange", alpha=0.4, label="Training data")
        ax.scatter(d.t_test, d.x_test_noisy, s=8, color="purple", alpha=0.4, label="Test data")
        ax.plot(d.t_full, d.x_full_true, color=TRUE_COLOR, linewidth=2, linestyle="--", label="True signal")
        ax.plot(d.t_full, run.predictions["full"], color=colors[name], linewidth=2, label="Prediction")
        ax.set_title(f"{name}  |  test RMSE = {run.metrics.loc['Test (vs true)', 'RMSE']:.3g}")
        ax.grid(alpha=0.5)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.show()


def plot_sweep_errors(runs):
    """Prediction error (vs true) of every run over the training + test windows, in one plot."""
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = _sweep_colors(runs)
    first = next(iter(runs.values()))
    _mark_test_window(ax, first)
    for name, run in runs.items():
        ax.plot(run.data.t_full, run.predictions["full"] - run.data.x_full_true, color=colors[name],
                linewidth=1.5, label=name)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pred - True")
    ax.set_title("Prediction error for each loss weighting")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_sweep_losses(runs):
    """Training curves of every run: data loss (vs noise floor) and unweighted physics loss."""
    fig, (ax_d, ax_p) = plt.subplots(1, 2, figsize=(16, 5))
    colors = _sweep_colors(runs)
    for name, run in runs.items():
        ax_d.plot(run.losses["data"], color=colors[name], linewidth=1.2, label=name)
        ax_p.plot(run.losses["physics"], color=colors[name], linewidth=1.2, label=name)
    ax_d.axhline(next(iter(runs.values())).noise_floor, color="black", linestyle=":", label="Noise floor")
    ax_d.set_title("Data loss (normalized)")
    ax_p.set_title("Physics loss (unweighted, training-window collocation points)")
    for ax in (ax_d, ax_p):
        ax.set_xlabel("Epoch")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    ax_p.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    plt.tight_layout()
    plt.show()
