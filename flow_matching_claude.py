"""
FLOW MATCHING IMPLEMENTATION - INTERVIEW STYLE
==============================================
Time Limit: 45 minutes
Difficulty: ⭐⭐⭐⭐☆

TASK: Implement Conditional Flow Matching (CFM) for generative modeling

MATHEMATICAL BACKGROUND:
Flow matching learns a vector field v_θ(x, t) that transforms a simple 
distribution (e.g., Gaussian) into a complex target distribution.

Key equation:
  dx/dt = v_θ(x, t)  where t ∈ [0, 1]

Training objective (Conditional Flow Matching):
  L(θ) = E[||v_θ(x_t, t) - u_t(x_t | x_1)||²]
  
where:
  - x_0 ~ N(0, I) (noise)
  - x_1 ~ p_data (data)
  - x_t = t·x_1 + (1-t)·x_0 (linear interpolation)
  - u_t(x_t | x_1) = (x_1 - x_0) (target velocity)

REQUIREMENTS:
1. Implement vector field network v_θ(x, t)
2. Implement forward sampling path: x_t = t·x_1 + (1-t)·x_0
3. Implement training loss
4. Implement ODE sampling using Euler method
5. Train on 2D toy dataset (e.g., mixture of Gaussians)

EVALUATION:
- Does the model learn to transform noise to data?
- Can you visualize the learned flow?
- Is the sampling process working correctly?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# PART 1: VECTOR FIELD NETWORK
# ============================================================================

class VectorField(nn.Module):
    """
    Neural network that predicts velocity v_θ(x, t)
    
    Input: 
      - x: position in space, shape (batch, dim)
      - t: time, shape (batch, 1) or (batch,)
    Output:
      - v: velocity, shape (batch, dim)
    """
    def __init__(self, dim=2, hidden_dim=128, time_embed_dim=32):
        super().__init__()
        self.dim = dim
        
        # TODO: Design network architecture
        # HINT: Common choices:
        # 1. Time embedding: map t to higher dimension
        # 2. Concatenate [x, time_emb]
        # 3. MLP with multiple layers
        # 4. Output should match input dimension
        
        # Example structure (you can modify):
        # self.time_embed = ...  # Time embedding layer
        # self.net = nn.Sequential(...)  # Main network
        
        self.time_embedding = nn.Sequential(
            nn.Linear(1,time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        input_dim = dim + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    
    def forward(self, x, t):
        """
        Args:
            x: (batch, dim) - position
            t: (batch,) or (batch, 1) - time in [0, 1]
        Returns:
            v: (batch, dim) - velocity
        """
        # TODO: Implement forward pass
        # 1. Handle time input shape
        # 2. Embed time
        # 3. Concatenate with x
        # 4. Pass through network
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        time_embed = self.time_embedding(t)
        x_and_t = torch.cat([x,time_embed],dim=-1)
        v = self.net(x_and_t)
        return v

# ============================================================================
# PART 2: FLOW MATCHING TRAINER
# ============================================================================

class ConditionalFlowMatching:
    """
    Implements Conditional Flow Matching training and sampling
    """
    def __init__(self, dim=2, hidden_dim=128, device='cpu'):
        self.dim = dim
        self.device = device
        
        # TODO: Initialize vector field network
        self.model = VectorField(self.dim, hidden_dim)  # YOUR CODE: VectorField(dim, hidden_dim).to(device)
    
    def sample_conditional_flow(self, x0, x1, t):
        """
        Sample points along the conditional flow path
        
        Path: x_t = (1 - t) * x_0 + t * x_1
        
        Args:
            x0: (batch, dim) - noise samples
            x1: (batch, dim) - data samples
            t: (batch, 1) - time points
        Returns:
            x_t: (batch, dim) - interpolated points
            u_t: (batch, dim) - target velocity (x_1 - x_0)
        """
        # TODO: Implement conditional flow sampling
        # 1. Compute x_t using linear interpolation
        # 2. Compute target velocity u_t
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x_t = (1-t)*x0 + t*x1
        u_t = x1 - x0
        return x_t, u_t
    
    def compute_loss(self, x1):
        """
        Compute Flow Matching loss
        
        Loss: E[||v_θ(x_t, t) - u_t||²]
        
        Args:
            x1: (batch, dim) - data samples
        Returns:
            loss: scalar
        """
        batch_size = x1.shape[0]
        
        # TODO: Implement training loss
        # 1. Sample noise x_0 ~ N(0, I)
        # 2. Sample time t ~ U[0, 1]
        # 3. Compute x_t and target velocity u_t
        # 4. Predict velocity v_θ(x_t, t)
        # 5. Compute MSE loss
        
        x0 = torch.randn(batch_size, self.dim)
        t = torch.rand(batch_size)
        x_t, u_t = self.sample_conditional_flow(x0,x1,t)
        v_t = self.model.forward(x_t, t)
        loss = F.mse_loss(v_t, u_t)
        return loss
        
    
    def sample(self, batch_size=100, num_steps=100):
        """
        Generate samples by solving ODE: dx/dt = v_θ(x, t)
        
        Use Euler method: x_{t+Δt} = x_t + Δt * v_θ(x_t, t)
        
        Args:
            batch_size: number of samples
            num_steps: number of integration steps
        Returns:
            samples: (batch_size, dim) - generated samples
        """
        # TODO: Implement ODE sampling
        # 1. Start from noise: x = torch.randn(batch_size, dim)
        # 2. Integrate from t=0 to t=1 using Euler method
        # 3. Return final samples
        
        x = torch.randn(batch_size, self.dim).to(self.device)
        delta_t = 1/num_steps
        t = torch.zeros(batch_size).to(x.device)
        for step in range(num_steps):
            x += delta_t * self.model.forward(x, t)
            t += delta_t
        return x

    
    def train_step(self, x1, optimizer):
        """
        Single training step
        
        Args:
            x1: (batch, dim) - batch of data
            optimizer: torch optimizer
        Returns:
            loss: scalar
        """
        optimizer.zero_grad()
        loss = self.compute_loss(x1)
        loss.backward()
        optimizer.step()
        return loss.item()


# ============================================================================
# PART 3: TOY DATASET
# ============================================================================

def create_toy_dataset(n_samples=10000, dataset_type='moons'):
    """
    Create 2D toy datasets for testing
    
    Args:
        n_samples: number of samples
        dataset_type: 'moons', 'circles', 'gaussians', 'spiral'
    Returns:
        data: (n_samples, 2) numpy array
    """
    if dataset_type == 'moons':
        # Two interleaving half circles
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=n_samples, noise=0.05)
        
    elif dataset_type == 'circles':
        # Two concentric circles
        from sklearn.datasets import make_circles
        data, _ = make_circles(n_samples=n_samples, noise=0.05, factor=0.5)
        
    elif dataset_type == 'gaussians':
        # Mixture of Gaussians
        n_per_component = n_samples // 4
        centers = [[-2, -2], [-2, 2], [2, -2], [2, 2]]
        data = []
        for center in centers:
            samples = np.random.randn(n_per_component, 2) * 0.5 + center
            data.append(samples)
        data = np.vstack(data)
        
    elif dataset_type == 'spiral':
        # Spiral pattern
        theta = np.linspace(0, 4*np.pi, n_samples)
        r = np.linspace(0.5, 3, n_samples)
        x = r * np.cos(theta) + np.random.randn(n_samples) * 0.1
        y = r * np.sin(theta) + np.random.randn(n_samples) * 0.1
        data = np.stack([x, y], axis=1)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return data.astype(np.float32)


# ============================================================================
# PART 4: VISUALIZATION
# ============================================================================

def visualize_flow(model, data, device='cpu', num_steps=20):
    """
    Visualize the learned flow field
    
    Args:
        model: trained ConditionalFlowMatching model
        data: original data points
        device: torch device
        num_steps: number of time steps to visualize
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Original data
    axes[0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=1)
    axes[0].set_title('Original Data')
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    
    # 2. Generated samples
    model.model.eval()
    with torch.no_grad():
        samples = model.sample(batch_size=1000, num_steps=100)
        samples = samples.cpu().numpy()
    axes[1].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
    axes[1].set_title('Generated Samples')
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    
    # 3. Flow trajectories
    with torch.no_grad():
        # Start from noise
        x = torch.randn(100, 2).to(device)
        trajectory = [x.cpu().numpy()]
        
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.ones(x.shape[0], 1).to(device) * (step * dt)
            v = model.model(x, t.squeeze())
            x = x + dt * v
            trajectory.append(x.cpu().numpy())
        
        trajectory = np.array(trajectory)  # (num_steps+1, 100, 2)
        
        for i in range(trajectory.shape[1]):
            axes[2].plot(trajectory[:, i, 0], trajectory[:, i, 1], 
                        'b-', alpha=0.3, linewidth=0.5)
        axes[2].scatter(trajectory[0, :, 0], trajectory[0, :, 1], 
                       c='green', s=10, label='Start (noise)')
        axes[2].scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], 
                       c='red', s=10, label='End (data)')
        axes[2].set_title('Flow Trajectories')
        axes[2].legend()
        axes[2].set_xlim(-4, 4)
        axes[2].set_ylim(-4, 4)
    
    # 4. Vector field
    x_grid = np.linspace(-4, 4, 20)
    y_grid = np.linspace(-4, 4, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    with torch.no_grad():
        t_vis = 0.5  # Visualize at t=0.5
        positions_torch = torch.FloatTensor(positions).to(device)
        t_torch = torch.ones(positions_torch.shape[0]).to(device) * t_vis
        velocities = model.model(positions_torch, t_torch).cpu().numpy()
    
    U = velocities[:, 0].reshape(X.shape)
    V = velocities[:, 1].reshape(X.shape)
    
    axes[3].quiver(X, Y, U, V, alpha=0.6)
    axes[3].set_title(f'Vector Field at t={t_vis}')
    axes[3].set_xlim(-4, 4)
    axes[3].set_ylim(-4, 4)
    
    plt.tight_layout()
    return fig


# ============================================================================
# PART 5: TRAINING LOOP
# ============================================================================

def train_flow_matching(dataset_type='moons', num_epochs=1000, batch_size=256):
    """
    Complete training pipeline
    
    Args:
        dataset_type: type of toy dataset
        num_epochs: number of training epochs
        batch_size: batch size
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"\nCreating {dataset_type} dataset...")
    data = create_toy_dataset(n_samples=10000, dataset_type=dataset_type)
    data_tensor = torch.FloatTensor(data).to(device)
    
    # Initialize model
    print("Initializing Flow Matching model...")
    cfm = ConditionalFlowMatching(dim=2, hidden_dim=128, device=device)
    optimizer = torch.optim.Adam(cfm.model.parameters(), lr=1e-3)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    losses = []
    
    for epoch in tqdm(range(num_epochs)):
        # Sample random batch
        indices = torch.randint(0, len(data_tensor), (batch_size,))
        batch = data_tensor[indices]
        
        # Training step
        loss = cfm.train_step(batch, optimizer)
        losses.append(loss)
        
        # Log progress
        if (epoch + 1) % 100 == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Final visualization
    print("\nGenerating visualizations...")
    fig = visualize_flow(cfm, data, device=device)
    plt.savefig('flow_matching_results.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'flow_matching_results.png'")
    
    # Plot loss curve
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    print("Saved loss curve to 'training_loss.png'")
    
    plt.show()
    
    return cfm, losses


# ============================================================================
# TEST SUITE
# ============================================================================

def test_vector_field():
    """Test vector field network"""
    print("Test 1: Vector Field Network")
    print("-" * 50)
    
    try:
        model = VectorField(dim=2, hidden_dim=64)
        x = torch.randn(10, 2)
        t = torch.rand(10)
        
        v = model(x, t)
        assert v.shape == (10, 2), f"Expected (10, 2), got {v.shape}"
        assert not torch.isnan(v).any(), "NaN in output"
        
        print(f"✓ Output shape: {v.shape}")
        print(f"✓ No NaN values")
        print("✓ Test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_conditional_flow():
    """Test conditional flow sampling"""
    print("Test 2: Conditional Flow Sampling")
    print("-" * 50)
    
    try:
        cfm = ConditionalFlowMatching(dim=2)
        x0 = torch.randn(10, 2)
        x1 = torch.randn(10, 2)
        t = torch.rand(10, 1)
        
        x_t, u_t = cfm.sample_conditional_flow(x0, x1, t)
        
        assert x_t.shape == (10, 2), f"x_t shape wrong: {x_t.shape}"
        assert u_t.shape == (10, 2), f"u_t shape wrong: {u_t.shape}"
        
        # Check interpolation property: x_t should be between x0 and x1
        # At t=0: x_t ≈ x0, at t=1: x_t ≈ x1
        
        print(f"✓ x_t shape: {x_t.shape}")
        print(f"✓ u_t shape: {u_t.shape}")
        print("✓ Test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def test_sampling():
    """Test ODE sampling"""
    print("Test 3: ODE Sampling")
    print("-" * 50)
    
    try:
        cfm = ConditionalFlowMatching(dim=2)
        samples = cfm.sample(batch_size=50, num_steps=10)
        
        assert samples.shape == (50, 2), f"Wrong shape: {samples.shape}"
        assert not torch.isnan(samples).any(), "NaN in samples"
        assert not torch.isinf(samples).any(), "Inf in samples"
        
        print(f"✓ Samples shape: {samples.shape}")
        print(f"✓ No NaN/Inf")
        print("✓ Test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("FLOW MATCHING - TEST SUITE")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Vector Field", test_vector_field()))
    results.append(("Conditional Flow", test_conditional_flow()))
    results.append(("ODE Sampling", test_sampling()))
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\nScore: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! Ready to train.")
    else:
        print(f"\n⚠️  Fix failing tests before training.")
    print("=" * 60)


# ============================================================================
# MATHEMATICAL INTUITION
# ============================================================================

def show_math_intuition():
    """Display mathematical concepts"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║              FLOW MATCHING - INTUITION                   ║
    ╚═══════════════════════════════════════════════════════════╝
    
    GOAL: Learn a vector field that transforms noise → data
    
    KEY CONCEPTS:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. FLOW: Continuous transformation over time            │
    │    dx/dt = v(x, t),  where t ∈ [0, 1]                  │
    │    x(0) = noise,  x(1) = data                          │
    │                                                         │
    │ 2. CONDITIONAL PATH: Linear interpolation               │
    │    x_t = (1-t)·x_0 + t·x_1                             │
    │    Target velocity: u_t = x_1 - x_0                    │
    │                                                         │
    │ 3. TRAINING: Match velocities                           │
    │    min E[||v_θ(x_t, t) - u_t||²]                       │
    │                                                         │
    │ 4. SAMPLING: Integrate ODE (Euler method)              │
    │    x_{t+Δt} = x_t + Δt · v_θ(x_t, t)                   │
    └─────────────────────────────────────────────────────────┘
    
    WHY DOES IT WORK?
    - Simple path: straight line from noise to data
    - Easy to compute target velocity
    - Network learns to push points along these paths
    - At inference: start from noise, follow learned flow
    
    VS. DIFFUSION MODELS:
    - Flow Matching: deterministic ODE, faster sampling
    - Diffusion: stochastic SDE, more sampling steps
    - Similar performance, different formulations
    """)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║        FLOW MATCHING IMPLEMENTATION                      ║
    ║           Conditional Flow Matching (CFM)                ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Your task: Implement Flow Matching for 2D generative modeling
    Time limit: 45 minutes
    
    Steps:
    1. Implement VectorField network
    2. Implement conditional flow sampling
    3. Implement training loss
    4. Implement ODE sampling
    5. Test on toy datasets
    
    Uncomment to start:
    """)
    
    # Uncomment these:
    #show_math_intuition()  # Review concepts
    run_all_tests()        # Test your implementation
    train_flow_matching(dataset_type='moons')  # Train the model