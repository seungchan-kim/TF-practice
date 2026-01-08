import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlowNetwork(nn.Module):
    """
    Learns v_theta(x_t, t | cond)
    """
    def __init__(self, state_dim, cond_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + cond_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ) #velocity model v

    def forward(self, x, t, cond):
        """
        x:    (B, state_dim)
        t:    (B, 1)
        cond: (B, cond_dim)
        """
        # TODO: concatenate x, t, cond correctly
        t = t.unsqueeze(-1)
        print("x", x.shape)
        print("t", t.shape)
        print("cond", cond.shape)
        inp = torch.cat((x,t,cond),1)
        return self.net(inp)


def interpolate(x0, x1, t):
    """
    x0: noise sample     (B, D)
    x1: data sample      (B, D)
    t:  time in [0,1]    (B, 1)

    Returns:
        x_t: interpolated state
    """
    # TODO: implement linear interpolation
    #from pdb import set_trace as bp; bp()
    t = t.unsqueeze(-1)
    x_t = (1-t) * x0 + t*x1
    return x_t

def ground_truth_flow(x0, x1):
    """
    For linear path:
        x_t = (1 - t) * x0 + t * x1

    dx_t / dt = x1 - x0
    """
    # TODO: return the true velocity field
    v = x1 - x0
    return v

def flow_matching_loss(model, x1, cond):
    """
    model: FlowNetwork
    x1:    data sample (B, D)
    cond:  conditioning variable (B, C)
    """
    B, D = x1.shape

    # 1. Sample noise
    # TODO
    x0 = torch.randn(B,D)

    # 2. Sample time uniformly in [0,1]
    # TODO
    t = torch.rand(B)

    # 3. Interpolate
    # TODO
    x_t = interpolate(x0,x1,t)

    # 4. Ground-truth flow
    # TODO
    v_gt = ground_truth_flow(x0,x1)

    # 5. Predicted flow
    # TODO
    v_pred = model.forward(x_t, t, cond)

    # 6. Loss (MSE)
    loss = F.mse_loss(v_pred, v_gt)
    return loss


def train_step(model, optimizer, x1, cond):
    optimizer.zero_grad()
    loss = flow_matching_loss(model, x1, cond)
    loss.backward()
    optimizer.step()
    return loss.item()



def main():
    torch.manual_seed(0)

    # Hyperparameters
    B = 128
    state_dim = 4
    cond_dim = 3
    lr = 1e-3
    steps = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy dataset
    x1 = torch.randn(B, state_dim, device=device)
    cond = torch.randn(B, cond_dim, device=device)

    # Model
    model = FlowNetwork(state_dim, cond_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    for step in range(steps):
        loss = train_step(model, optimizer, x1, cond)

        if step % 20 == 0:
            print(f"Step {step:03d} | Loss: {loss:.6f}")

    # Basic sanity checks
    with torch.no_grad():
        t_test = torch.rand(B, 1, device=device)
        x0_test = torch.randn(B, state_dim, device=device)
        x_t_test = interpolate(x0_test, x1, t_test)
        v_pred = model(x_t_test, t_test, cond)

        assert v_pred.shape == (B, state_dim)
        assert not torch.isnan(v_pred).any()

    print("✅ Test passed: forward, loss, and training are stable.")

if __name__ == "__main__":
    main()