"""Synthetic Simple Harmonic Oscillator data: time sampling, noisy signal and NN features."""
import numpy as np
import torch

# Sampling schemes accepted by `sample_time_points` (and their optional keyword arguments)
SAMPLING_SCHEMES = {
    "uniform":     {},
    "random":      {},
    "chebyshev":   {},
    "beta":        {"a": 0.5, "b": 0.5},
    "normal":      {"mu": None, "sigma": None},
    "exponential": {"scale": None, "reverse": False},
    "log":         {"t0": None},
    "jitter":      {"jitter": 0.3},
    "gaps":        {"n_windows": 3, "duty": 0.5},
    "clustered":   {"n_clusters": 8, "width": None},
    "custom":      {"density": None},
}


def sample_time_points(t_start, t_end, n_points, sampling="uniform", rng=None, **kw):
    """
    Return `n_points` sorted times in [t_start, t_end] following a sampling scheme.

    sampling options (optional keyword arguments and defaults in brackets):
      "uniform"      Evenly spaced grid (np.linspace).
      "random"       i.i.d. Uniform(t_start, t_end). Equivalent to the arrival times of a
                     homogeneous Poisson process conditioned on n_points events.
      "chebyshev"    Chebyshev-Lobatto nodes: dense at both ends, sparse in the middle.
      "beta"         Beta(a, b) rescaled to the interval [a=0.5, b=0.5].
                     a=b<1 -> both ends, a=b>1 -> centre, a<b -> start, a>b -> end.
      "normal"       Truncated Gaussian [mu=midpoint, sigma=span/6].
      "exponential"  Truncated exponential, dense at start [scale=span/3, reverse=False].
                     reverse=True makes it dense at the end.
      "log"          Log-spaced grid, dense at start, like scale-factor / redshift bins
                     in cosmology [t0=span*1e-3, offset that avoids log(0)].
      "jitter"       Regular cadence with Gaussian timing errors, like a scheduled
                     instrument [jitter=0.3, std as a fraction of the grid step].
      "gaps"         Observing seasons: samples only inside `n_windows` windows separated
                     by gaps (e.g. target visibility) [n_windows=3, duty=0.5, fraction observed].
      "clustered"    Observing nights: bursts of points around random epochs, like
                     ground-based photometry [n_clusters=8, width=span/100].
      "custom"       Inhomogeneous sampling from any density lambda(t) >= 0 via inverse CDF
                     [density=callable, required].
    """
    rng = np.random.default_rng() if rng is None else rng
    span = t_end - t_start
    # Treat explicit None as "use the default"
    kw = {k: v for k, v in kw.items() if v is not None}

    def truncated(draw):
        # Rejection sampling: keep drawing until n_points fall inside [t_start, t_end]
        out = np.empty(0)
        while out.size < n_points:
            s = draw(2 * n_points)
            out = np.concatenate([out, s[(s >= t_start) & (s <= t_end)]])
        return out[:n_points]

    if sampling == "uniform":
        t = np.linspace(t_start, t_end, n_points)

    elif sampling == "random":
        t = rng.uniform(t_start, t_end, n_points)

    elif sampling == "chebyshev":
        k = np.arange(n_points)
        t = t_start + span * (1 - np.cos(np.pi * k / (n_points - 1))) / 2

    elif sampling == "beta":
        t = t_start + span * rng.beta(kw.get("a", 0.5), kw.get("b", 0.5), n_points)

    elif sampling == "normal":
        mu, sigma = kw.get("mu", t_start + span / 2), kw.get("sigma", span / 6)
        t = truncated(lambda m: rng.normal(mu, sigma, m))

    elif sampling == "exponential":
        scale = kw.get("scale", span / 3)
        u = rng.uniform(size=n_points)
        x = -scale * np.log(1 - u * (1 - np.exp(-span / scale)))  # inverse CDF on [0, span]
        t = t_end - x if kw.get("reverse", False) else t_start + x

    elif sampling == "log":
        t0 = kw.get("t0", span * 1e-3)
        t = t_start + np.geomspace(t0, span + t0, n_points) - t0

    elif sampling == "jitter":
        grid = np.linspace(t_start, t_end, n_points)
        step = span / (n_points - 1)
        t = np.clip(grid + rng.normal(0, kw.get("jitter", 0.3) * step, n_points), t_start, t_end)

    elif sampling == "gaps":
        n_windows, duty = kw.get("n_windows", 3), kw.get("duty", 0.5)
        period = span / n_windows
        window = rng.integers(0, n_windows, n_points)
        t = t_start + window * period + rng.uniform(0, duty * period, n_points)

    elif sampling == "clustered":
        centres = rng.uniform(t_start, t_end, kw.get("n_clusters", 8))
        width = kw.get("width", span / 100)
        t = truncated(lambda m: rng.choice(centres, m) + rng.normal(0, width, m))

    elif sampling == "custom":
        grid = np.linspace(t_start, t_end, 10_000)
        cdf = np.cumsum(kw["density"](grid))
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
        t = np.interp(rng.uniform(size=n_points), cdf, grid)

    else:
        raise ValueError(f"Unknown sampling scheme: {sampling!r}. "
                         f"Options: {list(SAMPLING_SCHEMES)}")

    return np.sort(t)


def sho_solution(t, A, omega, phi=0.0):
    """Analytic SHO displacement x(t) = A cos(omega t + phi)."""
    return A * np.cos(omega * t + phi)


def sho_initial_conditions(A, omega, phi=0.0):
    """Analytic initial state (x(0), x'(0)) of the SHO."""
    return A * np.cos(phi), -A * omega * np.sin(phi)


def create_synthetic_HO_data(A, omega, phi=0.0, noise_std=None,
                             t_start=0.0, t_end=1.5, n_points=500, seed=42,
                             sampling="uniform", sampling_kwargs=None,
                             plot_noise_dist=False, plot_signal=False):
    """
    Generate synthetic Simple Harmonic Oscillator data x(t) = A cos(omega t + phi)
    with additive Gaussian noise, sampled at (possibly) non-uniform times.

    Parameters
    ----------
    A : float               Amplitude.
    omega : float           Angular frequency (rad/s).
    phi : float             Phase offset (rad).
    noise_std : float       Std dev of Gaussian noise (defaults to 0.3*A).
    t_start, t_end : float  Time interval.
    n_points : int          Number of time samples.
    seed : int              Random seed for reproducibility.
    sampling : str          Time sampling scheme, see `sample_time_points`.
    sampling_kwargs : dict  Extra keyword arguments for the sampling scheme.
    plot_noise_dist : bool  If True, plot the noise (error) distribution.
    plot_signal : bool      If True, plot the clean signal and noisy data.

    Returns
    -------
    t, x_clean, x_noisy : np.ndarray
    """
    if noise_std is None:
        noise_std = 0.3 * A

    # Time vector (separate RNG stream so the noise is the same for every sampling scheme)
    rng_t = np.random.default_rng(seed=[seed, 1])
    t = sample_time_points(t_start, t_end, n_points, sampling, rng=rng_t, **(sampling_kwargs or {}))

    # Clean displacement
    x_clean = sho_solution(t, A, omega, phi)

    # Add Gaussian noise
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=t.shape)
    x_noisy = x_clean + noise

    if plot_noise_dist or plot_signal:
        from .evaluation import plot_noise_distribution, plot_signal as _plot_signal
        if plot_noise_dist:
            plot_noise_distribution(noise)
        if plot_signal:
            _plot_signal(t, x_clean, x_noisy,
                         title=f"Synthetic Simple Harmonic Oscillator Data ({sampling} sampling)")

    return t, x_clean, x_noisy


def create_NN_features(t, x_noisy, normalize=True):
    """
    Build (optionally normalized) column-vector tensors to train the NN.

    Parameters
    ----------
    t : np.ndarray          Time samples.
    x_noisy : np.ndarray    Noisy displacement measurements.
    normalize : bool        If True, z-score normalize t and x using their mean/std.
                            If False, use mean=0 and std=1 (i.e. data is left as is).

    Returns
    -------
    t_norm, x_norm : torch.Tensor           Features and targets, shape (N, 1).
    t_mean, t_std, x_mean, x_std : torch.Tensor
        Statistics used, needed to normalize new inputs and unnormalize predictions.
    """
    # Convert data to tensors (reshape to column vectors)
    t_tensor = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    x_tensor = torch.tensor(x_noisy, dtype=torch.float32).reshape(-1, 1)

    if normalize:
        t_mean, t_std = t_tensor.mean(), t_tensor.std()
        x_mean, x_std = x_tensor.mean(), x_tensor.std()
    else:  # Un-normalization
        t_mean, t_std = torch.tensor(0.0), torch.tensor(1.0)
        x_mean, x_std = torch.tensor(0.0), torch.tensor(1.0)

    t_norm = (t_tensor - t_mean) / t_std
    x_norm = (x_tensor - x_mean) / x_std

    return t_norm, x_norm, t_mean, t_std, x_mean, x_std
