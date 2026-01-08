import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingLearner(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # v_theta(x_t, t, condition)

    def compute_loss(self, x_1, condition):
        """
        x_1: Real data (e.g., robot actions) [batch, dim]
        condition: Context (e.g., vision-language features) [batch, cond_dim]
        """
        batch_size = x_1.shape[0]
        
        # 1. Sample noise x_0 (Standard Gaussian)
        x_0 = torch.randn_like(x_1)
        
        # 2. Sample time t uniformly between 0 and 1
        # TODO: Implement t sampling [batch, 1]
        t = torch.rand(batch_size).unsqueeze(-1)
        
        # 3. Compute the probability path x_t (Interpolation)
        # For Optimal Transport, path is a straight line: x_t = (1 - t) * x_0 + t * x_1
        # TODO: Implement x_t
        #t shape = batch,1
        #x1 = batch, dim
        x_t = (1-t) * x_0  + x_1 * t
        
        # 4. Compute the target velocity (v_t)
        # For the straight line path, velocity is constant: v_t = x_1 - x_0
        # TODO: Implement target_v
        target_v = x_1 - x_0
        
        # 5. Predict velocity using the model
        # The model takes (x_t, t, condition) and predicts the velocity vector
        pred_v = self.model(x_t, t, condition)
        
        # 6. Compute Mean Squared Error Loss
        # TODO: Implement loss
        loss = F.mse_loss(pred_v, target_v)
        
        return loss

    @torch.no_grad()
    def sample(self, condition, num_steps=10):
        """
        Inference using Euler integration to solve the ODE: dx = v_theta(x, t, c) dt
        """
        # Start from pure noise x_0
        # TODO: Initialize x with random noise
        x = torch.randn(batch_size, input_dim)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            # Current time t
            t = torch.ones((x.shape[0], 1), device=x.device) * (i / num_steps)
            
            # 1. Predict velocity v
            # TODO: Get v from the model
            v = self.model(x, t, condition)
            
            # 2. Update x using Euler step: x = x + v * dt
            # TODO: Implement update
            x = x + v*dt
            
        return x # This should now be x_1 (predicted action)

# --- 연습용 MLP 모델 구조 ---
class VelocityModel(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        # t를 입력받기 위해 input_dim + 1, 그리고 condition 결합
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1 + cond_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, t, c):
        # x: [B, D], t: [B, 1], c: [B, C]
        inputs = torch.cat([x, t, c], dim=-1)
        return self.net(inputs)


if __name__ == "__main__":
    # 하이퍼파라미터 설정
    batch_size = 16
    input_dim = 2  # 예: 로봇의 (x, y) 좌표 액션
    cond_dim = 10  # 예: 비전-언어 피처 차원
    
    # 모델 및 러너 초기화
    model = VelocityModel(input_dim, cond_dim)
    learner = FlowMatchingLearner(model)
    
    # 가짜 데이터 생성 (x_1: 실제 액션, c: 컨디션)
    x_1 = torch.randn((batch_size, input_dim))
    c = torch.randn((batch_size, cond_dim))
    
    print("--- Testing Training Step ---")
    loss = learner.compute_loss(x_1, c)
    print(f"Loss: {loss.item():.4f}")
    
    # Loss가 정상적으로 스칼라 값으로 나오는지 확인
    assert loss.dim() == 0, "Loss should be a scalar!"
    print("Training Step Test Passed!\n")
    
    print("--- Testing Sampling Step ---")
    num_steps = 50
    sampled_actions = learner.sample(c, num_steps=num_steps)
    
    # 결과 차원 확인
    print(f"Sampled actions shape: {sampled_actions.shape}")
    assert sampled_actions.shape == (batch_size, input_dim), "Sampled shape mismatch!"
    
    # 값이 nan이 아닌지 확인
    assert not torch.isnan(sampled_actions).any(), "Sampled actions contain NaNs!"
    print(f"Sampling Step Test Passed! (used {num_steps} Euler steps)")
    
    print("\n[Final Result] All tests passed successfully!")