"""Network architecture, physics-informed loss terms and training loops (ANN and PINN)."""
import torch
import torch.nn as nn


class SHONet(nn.Module):
    def __init__(self, hidden_sizes=(32, 64, 128, 64, 32)):
        """hidden_sizes: number of neurons (out_features) of each hidden layer, in order."""
        super().__init__()
        layers = []
        in_features = 1
        for out_features in hidden_sizes:
            layers += [nn.Linear(in_features, out_features), nn.Tanh()]
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_model(hidden_sizes, seed=0):
    """Create a SHONet with reproducible initial weights (same seed -> same weights)."""
    torch.manual_seed(seed)
    return SHONet(hidden_sizes=hidden_sizes)


# ---------------------------------------------------------------------------
# Physics-informed components
# ---------------------------------------------------------------------------
def physics_residual(model, t_norm, omega, t_std, x_std, x_mean=0.0):
    """
    Computes x'' + omega^2 * x = 0
    correcting for the chain rule introduced by normalization.
    `t_norm` must have requires_grad=True.
    """
    x_norm = model(t_norm)

    # dx_norm/dt_norm via autograd
    dx_norm = torch.autograd.grad(x_norm, t_norm, grad_outputs=torch.ones_like(x_norm),
                                  create_graph=True)[0]
    d2x_norm = torch.autograd.grad(dx_norm, t_norm, grad_outputs=torch.ones_like(dx_norm),
                                   create_graph=True)[0]

    # Chain rule: since t_norm = (t - t_mean)/t_std and x_norm = (x - x_mean)/x_std,
    # d2x/dt2 = (x_std / t_std^2) * d2x_norm/dt_norm2
    d2x_dt2 = (x_std / t_std**2) * d2x_norm
    x_real = x_norm * x_std + x_mean

    residual = d2x_dt2 + (omega**2) * x_real
    return residual


def ic_loss(model, t_mean, t_std, x_mean, x_std, x0, v0, t0=0.0):
    """
    Enforces x(t0) = x0 and x'(t0) = v0
    """
    t0 = torch.tensor([[t0]], dtype=torch.float32, requires_grad=True)
    t0_norm = (t0 - t_mean) / t_std

    x0_pred_norm = model(t0_norm)
    x0_pred_real = x0_pred_norm * x_std + x_mean

    dx0_norm = torch.autograd.grad(x0_pred_norm, t0_norm,
                                   grad_outputs=torch.ones_like(x0_pred_norm),
                                   create_graph=True)[0]
    # chain rule: dx/dt = (x_std/t_std) * dx_norm/dt_norm
    v0_pred_real = (x_std / t_std) * dx0_norm

    loss_x0 = (x0_pred_real - x0)**2
    loss_v0 = (v0_pred_real - v0)**2

    return loss_x0.squeeze() + loss_v0.squeeze()


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------
def train_ANN(model, t_data_norm, x_data_norm, lr=1e-3, n_epochs=1500, print_every=200):
    """
    Train a plain ANN with the data MSE only (normalized space).

    Returns
    -------
    losses : list  Data loss per epoch.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(t_data_norm)
        loss = criterion(pred, x_data_norm)   # compare in normalized space
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if print_every and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")

    return losses


def train_PINN(model, t_data_norm, x_data_norm, t_colloc_norm,
               omega, t_mean, t_std, x_mean, x_std, x0, v0,
               lambda_physics=5e-4, lambda_ic=5e-4, lr=5e-5, n_epochs=int(2e4), print_every=500):
    """
    Train a PINN with loss = data loss + lambda_physics * physics loss + lambda_ic * IC loss.

    Returns
    -------
    losses : dict of lists  'data', 'physics', 'ic', 'total' loss per epoch
             (physics and IC are stored unweighted).
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = {"data": [], "physics": [], "ic": [], "total": []}

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Data loss (normalized space)
        x_pred_norm = model(t_data_norm)
        loss_data = criterion(x_pred_norm, x_data_norm)

        # Physics loss (ODE residual on collocation points)
        residual = physics_residual(model, t_colloc_norm, omega, t_std, x_std, x_mean)
        loss_physics = torch.mean(residual**2)

        # Initial condition loss
        loss_ic = ic_loss(model, t_mean, t_std, x_mean, x_std, x0, v0)

        loss = loss_data + lambda_physics * loss_physics + lambda_ic * loss_ic
        loss.backward()
        optimizer.step()

        losses["data"].append(loss_data.item())
        losses["physics"].append(loss_physics.item())
        losses["ic"].append(loss_ic.item())
        losses["total"].append(loss.item())

        if print_every and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Total: {loss.item():.5f} | "
                  f"Data: {loss_data.item():.5f} | Physics: {loss_physics.item():.5f} | "
                  f"IC: {loss_ic.item():.5f}")

    return losses
