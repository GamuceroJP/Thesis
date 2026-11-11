"""Governing ODE of the (damped, driven) harmonic oscillator and its exact solution."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Oscillator:
    """
    Linear oscillator with unit mass:

        x'' + 2 zeta omega x' + omega^2 x = F0 cos(Omega t)

    omega : float  Natural angular frequency (rad/s).
    zeta  : float  Damping ratio (0 <= zeta < 1, under-damped).
    F0    : float  Driving amplitude (force per unit mass).
    Omega : float  Driving angular frequency (rad/s).

    zeta = F0 = 0 is the simple harmonic oscillator, F0 = 0 the damped one.
    """
    omega: float
    zeta: float = 0.0
    F0: float = 0.0
    Omega: float = 0.0

    def __post_init__(self):
        if not 0 <= self.zeta < 1:
            raise ValueError(f"Only the under-damped regime 0 <= zeta < 1 is supported, got {self.zeta}")
        if self.F0 and self.zeta == 0 and self.Omega == self.omega:
            raise ValueError("Undamped resonance (zeta = 0, Omega = omega) has no bounded solution")

    @property
    def name(self):
        if self.F0:
            return "Driven Damped Harmonic Oscillator" if self.zeta else "Driven Harmonic Oscillator"
        return "Damped Harmonic Oscillator" if self.zeta else "Simple Harmonic Oscillator"

    @property
    def residual_label(self):
        """LaTeX of |ODE residual|, for plot labels."""
        lhs = "\\ddot{x}"
        if self.zeta:
            lhs += " + 2\\zeta\\omega\\dot{x}"
        lhs += " + \\omega^2 x"
        if self.F0:
            lhs += " - F_0\\cos(\\Omega t)"
        return f"$|{lhs}|$"

    @property
    def omega_d(self):
        """Damped angular frequency of the transient."""
        return self.omega * np.sqrt(1 - self.zeta**2)

    # ---- ODE -------------------------------------------------------------
    def forcing(self, t):
        """F0 cos(Omega t), for np.ndarray or torch.Tensor t."""
        cos = torch.cos if torch.is_tensor(t) else np.cos
        return self.F0 * cos(self.Omega * t)

    def residual(self, x, dx, d2x, t):
        """x'' + 2 zeta omega x' + omega^2 x - F0 cos(Omega t), in physical units."""
        return d2x + 2 * self.zeta * self.omega * dx + self.omega**2 * x - self.forcing(t)

    # ---- Exact solution --------------------------------------------------
    def steady_state(self):
        """Amplitude G and phase lag delta of the particular solution G cos(Omega t - delta)."""
        if not self.F0:
            return 0.0, 0.0
        a, b = self.omega**2 - self.Omega**2, 2 * self.zeta * self.omega * self.Omega
        return self.F0 / np.hypot(a, b), np.arctan2(b, a)

    def _coefficients(self, x0, v0):
        # x(t) = exp(-zeta omega t) (C1 cos(omega_d t) + C2 sin(omega_d t)) + G cos(Omega t - delta)
        G, delta = self.steady_state()
        C1 = x0 - G * np.cos(delta)
        C2 = (v0 - G * self.Omega * np.sin(delta) + self.zeta * self.omega * C1) / self.omega_d
        return C1, C2, G, delta

    def solution(self, t, x0, v0):
        """Exact x(t) for initial conditions x(0) = x0, x'(0) = v0."""
        C1, C2, G, delta = self._coefficients(x0, v0)
        wd, decay = self.omega_d, np.exp(-self.zeta * self.omega * t)
        return decay * (C1 * np.cos(wd * t) + C2 * np.sin(wd * t)) + G * np.cos(self.Omega * t - delta)

    def velocity(self, t, x0, v0):
        """Exact x'(t) for initial conditions x(0) = x0, x'(0) = v0."""
        C1, C2, G, delta = self._coefficients(x0, v0)
        wd, k, decay = self.omega_d, self.zeta * self.omega, np.exp(-self.zeta * self.omega * t)
        transient = (-k * C1 + wd * C2) * np.cos(wd * t) + (-k * C2 - wd * C1) * np.sin(wd * t)
        return decay * transient - G * self.Omega * np.sin(self.Omega * t - delta)
