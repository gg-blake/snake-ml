import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features, bias=False),
            nn.ReLU()
        )

    def forward(self, inputs):
        # Implement forward pass logic
        # Input -> Hidden Layer
        logits = self.ffwd(inputs)

        return logits

class NDVector:
    def __init__(self, n_dims=2):
        self.n_dims = n_dims
        self.dim_ids = [[i, i+n_dims] for i in range(0, n_dims)]
        
        _plane_ids = set([])
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                
                _plane_ids.add(frozenset([i, j]))
        
        self.plane_ids = torch.stack([torch.tensor(list(i)) for i in _plane_ids])
        
        self.normals = torch.eye(n_dims)
        
    def rotate(self, angle: torch.Tensor, precision: float=1e1):
        basis = self.normals[self.plane_ids].transpose(-1, -2)
        Q, _ = torch.linalg.qr(basis)
        #Q = basis
        m = Q.shape[2]
        n = Q.shape[1]
        
        v_projected = torch.matmul(Q.transpose(-1, -2),  self.normals)
        
        cos_theta = torch.round(torch.cos(torch.tensor(angle))*precision)/precision
        sin_theta = torch.round(torch.sin(torch.tensor(angle))*precision)/precision
        rotation_matrix = torch.eye(m).unsqueeze(0).repeat(angle.shape[0], 1, 1)
        
        rotation_matrix[:, 0, 0] = cos_theta
        rotation_matrix[:, 0, 1] = -sin_theta
        rotation_matrix[:, 1, 0] = sin_theta
        rotation_matrix[:, 1, 1] = cos_theta
        
        v_rotated_projected = torch.matmul(rotation_matrix, v_projected)
        v_rotated = torch.matmul(Q, v_rotated_projected)
        
        updated_normals = F.normalize(v_rotated + self.normals.unsqueeze(0).repeat(self.plane_ids.shape[0], 1, 1), dim=-1)
        updated_normals = F.normalize(torch.sum(updated_normals, dim=0), dim=-1)
        self.normals = torch.round(updated_normals*precision)/precision
        
    def rotate_discrete_left(self, dim):
        expanded_normals = self.normals.repeat(2, 1)
        expanded_normals[self.n_dims:] *= -1
        expanded_plane_ids = self.plane_ids.repeat(1, 2)
        expanded_plane_ids[:, 2:] = expanded_plane_ids[:, 2:] + (torch.ones((((self.n_dims - 1) * self.n_dims) // 2, 2,)) * self.n_dims)
        tmp = expanded_normals[expanded_plane_ids[dim][0]]
        for i in range(3):
            expanded_normals[expanded_plane_ids[dim][i]] = expanded_normals[expanded_plane_ids[dim][i+1]]
            
        expanded_normals[expanded_plane_ids[dim][-1]] = tmp
        self.normals = expanded_normals[:self.n_dims]
        
    def rotate_discrete_right(self, dim):
       for i in range(3):
           self.rotate_discrete_left(dim)
        
class GameObject:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.pos = torch.zeros(n_dims, dtype=torch.float32)
        self.vel = NDVector(n_dims)
        
    def facing(self, pos: torch.Tensor):
        assert pos.shape[0] == self.n_dims
        norms = self.vel.normals.repeat(2, 1)
        norms[self.n_dims:] *= -1
        return F.cosine_similarity((self.pos - pos).repeat(self.n_dims * 2, 1), norms, dim=1)
        
    def batch_facing(self, pos: torch.Tensor):
        # TODO: Vectorize this method
        return torch.stack([self.facing(p) for p in pos])
        
    def distance_from_bounds(self, scale: int):
        norms = self.vel.normals.repeat(2, 1)
        norms[self.n_dims:] *= -1
        return torch.sum(torch.abs(scale * norms) - torch.abs(self.pos * norms), dim=-1)
        
    def batch_distance_from_bounds(self, pos: torch.Tensor):
        return torch.cdist(self.pos.unsqueeze(0), pos.float(), p=2)
        

if __name__ == '__main__':
    food = torch.tensor([1, 9, 0], dtype=torch.float32)
    obj = GameObject(3)
    vec = NDVector(3)
    #vec.turn_left_angle(0, 0.34)
    print(list(vec.normals.values()))