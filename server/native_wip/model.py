import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from time import time, sleep
import onnxruntime as ort

if torch.cuda.is_available():
    print("GPU found")
    torch.cuda.set_device(0)
    sleep(2)

TORCH_LOGS="dynamic"

class NDVector(nn.Module):
    def __init__(self, n_dims: int):
        super(NDVector, self).__init__()
        _plane_ids = set([])
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                
                _plane_ids.add(frozenset([i, j]))
        
        _plane_ids = torch.stack([torch.tensor(list(i)) for i in _plane_ids])
        
        v = torch.eye(n_dims)
        
        # Register buffers for constants
        self.register_buffer('dim_ids', torch.stack([torch.tensor([i, i+n_dims]) for i in range(0, n_dims)]))
        self.register_buffer('plane_ids', _plane_ids)
        self.register_buffer('normal_vectors', v)
        self.register_buffer('n_dims', torch.tensor(n_dims))
        
    def forward(self, angle: torch.Tensor, X: torch.Tensor):
        basis = X[self.plane_ids].transpose(-1, -2)
        # Q, _ = torch.linalg.qr(basis)
        Q = basis
        m = Q.shape[2]
        n = Q.shape[1]
        
        v_projected = torch.matmul(Q.transpose(-1, -2),  X)
        
        cos_theta = torch.cos(torch.tensor(angle))
        sin_theta = torch.sin(torch.tensor(angle))
        rotation_matrix = torch.eye(m).unsqueeze(0).repeat(angle.shape[0], 1, 1)
        
        rotation_matrix[:, 0, 0] = cos_theta
        rotation_matrix[:, 0, 1] = -sin_theta
        rotation_matrix[:, 1, 0] = sin_theta
        rotation_matrix[:, 1, 1] = cos_theta
        
        v_rotated_projected = torch.matmul(rotation_matrix, v_projected)
        v_rotated = torch.matmul(Q, v_rotated_projected)
        
        updated_normals = F.normalize(v_rotated + X.unsqueeze(0).repeat(self.plane_ids.shape[0], 1, 1), dim=-1)
        updated_normals = F.normalize(torch.sum(updated_normals, dim=0), dim=-1)
        
        return updated_normals
    
class Model(nn.Module):
    def __init__(self, n_dims: int, _hidden_features: int):
        super(Model, self).__init__()
        self.nd_vector = NDVector(n_dims)
        
        _in_features = n_dims * 6 - 2
        _out_features = self.nd_vector.plane_ids.shape[0]
        
        self.ffwd = nn.Sequential(
            nn.Linear(_in_features, _hidden_features, bias=False),
            nn.ReLU(),
            nn.Linear(_hidden_features, _out_features, bias=False),
            nn.ReLU()
        )
        
        self.register_buffer('in_features', torch.tensor(_in_features))
        self.register_buffer('hidden_features', torch.tensor(_hidden_features))
        self.register_buffer('out_features', torch.tensor(_out_features))
        self.register_buffer('n_dims', torch.tensor(n_dims))
        
    def forward(self, snake_pos: torch.Tensor, snake_history: torch.Tensor, food_pos: torch.Tensor, bounds: torch.Tensor, vel: torch.Tensor):
        # Extend basis vectors to negative axis
        signed_vel = vel.repeat(2, 1)
        signed_vel[self.n_dims:] *= -1
        
        # Extend unit vectors to negative axis
        signed_unit_vectors = self.nd_vector.normal_vectors.repeat(2, 1)
        signed_unit_vectors[:self.n_dims] *= -1
        
        # Food data processing and vectorizing (n_dims * 2)
        food_norms = F.cosine_similarity((snake_pos - food_pos).repeat(self.n_dims * 2, 1), signed_vel, dim=1)
        nearby_food = torch.clamp(food_norms, min=-1, max=1)
        food_dist = (snake_pos - food_pos).pow(2).sum(-1).sqrt()
        
        # Body data processing and vectorizing (5) (n_dims * 2 - 1)
        body_norms = signed_vel + snake_pos.repeat(self.n_dims * 2, 1)
        body_dist = torch.cdist(body_norms, snake_history[:-1])
        nearby_body = torch.any(torch.cat([body_dist[:self.n_dims], body_dist[self.n_dims+1:]]) == 0, dim=1).float()
        
        def calculate_bounds(bound_tensor: torch.Tensor) -> torch.Tensor:
            # Calculate the euclidean distance of the snake to the game boundary
            bounds_dist = (bounds * signed_unit_vectors)  - (snake_pos * torch.abs(signed_unit_vectors))
            bounds_pos = torch.sum(bounds_dist[:self.n_dims], dim=-1)
            bounds_neg = torch.sum(bounds_dist[self.n_dims:], dim=-1)
            
            # Project the pos and neg distance vectors to the snake's velocity vectors, treating the velocity vectors as basis vectors
            u = vel / torch.norm(vel, dim=1, keepdim=True)
            dot_products_pos = torch.matmul(u, bounds_pos)
            dot_products_neg = torch.matmul(u, bounds_neg)
            proj_pos = torch.sum(dot_products_pos.unsqueeze(1) * u, dim=-1)
            proj_neg = torch.sum(dot_products_neg.unsqueeze(1) * u, dim=-1)
            
            return torch.cat([proj_pos, proj_neg])
        
        # Bounds data processing and vectorizing (3) (n_dims * 2 - 1)
        nearby_bounds = (calculate_bounds(bounds - 1) == 0).float()
        nearby_bounds = torch.cat([nearby_bounds[:self.n_dims], nearby_bounds[self.n_dims+1:]])
        
        inputs = torch.cat([nearby_food, nearby_bounds, nearby_body])
        
        logits = 2 * (F.softmax(10 * self.ffwd(inputs)) - 0.5) * (torch.pi / 2) # Shift to interval [-π/2, π/2]
        
        # NOTE: We will probably need to mask rotation for rotation on the plane perpendicular to the forward vector
        new_vec = self.nd_vector(logits, vel)
        
        return snake_pos + new_vec[0], new_vec, food_norms, food_dist, body_dist, calculate_bounds(bounds) 
    
def export():
    model = Model(3, 20)
    food_pos = torch.tensor([-9, 18, 3], dtype=torch.float32)
    snake_pos = torch.tensor([-1, 6, -1], dtype=torch.float32)
    snake_history = torch.randint(-10, 10, (8,3,), dtype=torch.float32)
    bounds = torch.tensor([-1, 6, 10])
    torch.onnx.export(
        model, 
        (snake_pos, snake_history, food_pos, bounds, model.nd_vector.normal_vectors.float()), 
        "3D-snake.onnx", 
        export_params=True, 
        opset_version=12, 
        do_constant_folding=True, 
        input_names=['snake_pos', 'snake_history', 'food_pos', 'bounds', 'vel'], 
        output_names=['next_position', 'velocity', 'food_norms', 'food_dist', 'body_dist', 'bounds_dist'], 
        dynamic_axes={
            'snake_pos': {0: 'batch_size'},
            'snake_history': {0: 'sequence_length'},
            'food_pos': {0: 'batch_size'},
            'bounds': {0: 'batch_size'}
        }
    )

def onnx_import():
    # Load the ONNX model
    ort_session = ort.InferenceSession("3D-snake.onnx")

    # Prepare dynamic inputs as PyTorch tensors
    snake_pos = torch.randn(3)  # batch_size = 3, 4 dimensions
    snake_history = torch.randn(18, 3)  # history_length = 10, 4 dimensions
    food_pos = torch.randn(3)  # batch_size = 3, 4 dimensions
    bounds = torch.tensor([10, 10, 10], dtype=torch.int64)  # batch_size = 3
    vel = torch.eye(3)  # batch_size = 3, 3 dimensions

    # Convert PyTorch tensors to NumPy for ONNX Runtime
    inputs = {
        'snake_pos': snake_pos.numpy(), 
        'snake_history': snake_history.numpy(), 
        'food_pos': food_pos.numpy(), 
        'bounds': bounds.numpy(), 
        'vel': vel.numpy()
    }

    # Run inference with ONNX Runtime
    start = time()
    outputs = ort_session.run(None, inputs)
    end = time()
    print(f"Completed in {end - start}s")

    # Convert the ONNX Runtime outputs back to PyTorch tensors
    next_position = torch.from_numpy(outputs[0])
    velocity = torch.from_numpy(outputs[1])

    # Print the results
    print("Next position (PyTorch Tensor):", next_position)
    print("Velocity (PyTorch Tensor):", velocity)

def test():
    m = Model(3, 20)
    food_pos = torch.tensor([-9, 18, 3], dtype=torch.float32)
    snake_pos = torch.tensor([-6, -8, 1], dtype=torch.float32)
    snake_history = torch.randint(-10, 10, (8,3,), dtype=torch.float32)
    bounds = torch.tensor([7, 7, 7])
    steps = 10000
    vel = NDVector(3)
    time_a = time()
    next_position, vel = m(snake_pos, snake_history, food_pos, bounds, vel.normal_vectors.float())
    for i in range(steps):
        next_position, vel = m(next_position, snake_history, food_pos, bounds, vel)
        #os.system('cls')
        #print(f"Steps: {i}\n\nCurrent Position: {next_position}\n\nCurrent Velocity: {vel}")
    
    time_b = time()
    print(f"Completed in {time_b - time_a}\nAverage Forward Pass Time: {(time_b - time_a) / steps}")

if __name__ == '__main__':
    export()
