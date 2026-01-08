import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.datasets import make_moons

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# 1. Dataset: 2D moons
# -------------------------

class MoonsDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int = 10_000, noise: float = 0.1):
        X, _ = make_moons(n_samples=n_samples, noise=noise)
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Return a single 2D data point x_1 ~ p_data
        return self.X[idx]


# -------------------------
# 2. Time-dependent MLP
#    v_theta(t, x_t)
# -------------------------

class TimeEmbedding(nn.Module):
    """Simple sinusoidal time embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B, 1) in [0, 1]
        return: (B, dim)
        """
        # TODO: implement a simple sinusoidal embedding of t
        # hint: use frequencies like 1, 2, 4, 8, ...
        # and concat sin, cos
        # shape after embedding: (B, dim)
        t = t.view(-1)  # (B,)

        # half of dimensions will be sin, half cos
        half_dim = self.dim // 2
        # frequencies: 1, 2, 4, 8, ... (size = half_dim)
        freqs = 2.0 ** torch.arange(half_dim, device=t.device, dtype=t.dtype)  # (half_dim,)

        # outer product: (B, 1) * (1, half_dim) -> (B, half_dim)
        angles = t[:, None] * freqs[None, :]

        emb_sin = torch.sin(angles)  # (B, half_dim)
        emb_cos = torch.cos(angles)  # (B, half_dim)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (B, 2 * half_dim)

        # if dim is odd, pad one zero dim
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(emb.shape[0], self.dim - emb.shape[-1],
                              device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=-1)

        return emb


class VectorField(nn.Module):
    """
    Simple MLP that takes (x_t, t) and predicts velocity v_theta(t, x_t) in R^2.
    """
    def __init__(self, hidden_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)

        in_dim = 2 + time_dim  # x_t (2D) + time embedding
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 2)
        t:   (B, 1)
        return: v_theta(t, x_t) of shape (B, 2)
        """
        t_emb = self.time_embedding(t)  # (B, time_dim)
        h = torch.cat([x_t, t_emb], dim=-1)
        v = self.net(h)
        return v


# -------------------------
# 3. Flow Matching Objective
# -------------------------
# We use the simple linear interpolant:
#   x_t = (1 - t) * x_0 + t * x_1
# with x_0 ~ N(0, I), x_1 ~ p_data (moons).
#
# For this path, a common choice of target velocity is:
#   v_target(t, x_0, x_1) = x_1 - x_0
# which corresponds to the conditional OT path in a basic setting.[web:11]

def sample_prior_like(x_data: torch.Tensor) -> torch.Tensor:
    """
    Sample x_0 ~ N(0, I) with the same shape as x_data.
    """
    return torch.randn_like(x_data)


def interpolate_linear(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation between x0 and x1 at time t in [0, 1].
    x0, x1: (B, 2)
    t:      (B, 1)
    """
    # TODO: implement x_t = (1 - t) * x0 + t * x1
    x_t = (1-t) * x0 + t * x1
    return x_t


def get_target_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Target velocity for linear / OT path.
    For the simple case, use v_target = x1 - x0 (independent of t).[web:11]
    """
    # TODO: return x1 - x0
    return x1 - x0


def flow_matching_loss(model: nn.Module,
                       x1: torch.Tensor,
                       device: torch.device) -> torch.Tensor:
    """
    Compute the flow matching loss for a batch.[web:9][web:11]
    x1: (B, 2) data samples ~ p_data
    """
    B = x1.shape[0]

    # 1) sample t ~ Uniform(0, 1) for each sample
    #    shape (B, 1)
    # TODO: implement
    # hint: use torch.rand(B, 1, device=device)
    t = torch.rand(B,1)

    # 2) sample x0 ~ N(0, I) with same shape as x1
    x0 = sample_prior_like(x1)

    # 3) compute x_t via linear interpolation
    x_t = interpolate_linear(x0, x1, t)

    # 4) compute target velocity v_target
    v_target = get_target_velocity(x0, x1)

    # 5) predict v_theta(t, x_t)
    v_pred = model(x_t, t)

    # 6) MSE loss between v_pred and v_target
    # TODO: implement MSE loss (mean over batch and dimensions)
    loss = F.mse_loss(v_pred, v_target)

    return loss


# -------------------------
# 4. Training Loop
# -------------------------

def train_flow_matching(
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    dataset = MoonsDataset(n_samples=10_000, noise=0.1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VectorField(hidden_dim=128, time_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            x1 = batch.to(DEVICE)  # (B, 2)

            # TODO:
            # 1) compute flow matching loss
            # 2) backprop and optimizer step
            # 3) accumulate running_loss (sum)

            loss = flow_matching_loss(model, x1, x1.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # backprop
            # ...
            running_loss += loss
        # normalize running_loss by number of samples
        running_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss:.4f}")

    return model


if __name__ == "__main__":
    train_flow_matching()
