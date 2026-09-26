"""Simple Harmonic Oscillator: plain ANN vs PINN experiments."""
from .data import (SAMPLING_SCHEMES, sample_time_points, sho_solution, sho_initial_conditions,
                   create_synthetic_HO_data, create_NN_features)
from .models import SHONet, build_model, physics_residual, ic_loss, train_ANN, train_PINN
from .evaluation import (predict, ode_residual, compute_metrics, metrics_table,
                         plot_noise_distribution, plot_signal, plot_fit, plot_sampling_schemes,
                         plot_ann_loss, plot_pinn_losses, plot_model_fit,
                         plot_data_loss_vs_noise_floor, plot_predictions_full, plot_error_full,
                         plot_ode_residual, plot_predictions_grid, plot_summary_bars)
from .experiment import (SHOConfig, SHODataset, ExperimentResult, generate_data, run_experiment,
                         compare_sampling_schemes, summarize)
