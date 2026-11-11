"""Harmonic oscillators (simple, damped, driven): plain ANN vs PINN experiments."""
from .physics import Oscillator
from .data import (SAMPLING_SCHEMES, sample_time_points, sho_solution, sho_initial_conditions,
                   create_synthetic_data, create_synthetic_HO_data, create_NN_features)
from .models import SHONet, build_model, physics_residual, ic_loss, train_ANN, train_PINN
from .evaluation import (predict, predict_velocity, ode_residual, compute_metrics, metrics_table,
                         plot_noise_distribution, plot_signal, plot_fit, plot_sampling_schemes,
                         plot_ann_loss, plot_pinn_losses, plot_model_fit,
                         plot_data_loss_vs_noise_floor, plot_predictions_full, plot_error_full,
                         plot_ode_residual, plot_phase_portrait, plot_power_spectrum,
                         plot_predictions_grid, plot_summary_bars,
                         plot_lambda_sweep, plot_sweep_predictions, plot_sweep_errors, plot_sweep_losses)
from .experiment import (SHOConfig, SHODataset, ExperimentResult, PINNRun, generate_data, run_experiment,
                         run_pinn, sweep_pinn, summarize_runs, compare_sampling_schemes, summarize)
