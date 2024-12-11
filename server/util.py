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

    def backward(self, outputs, targets):
        # Implement backward pass logic
        loss = F.cross_entropy(outputs, targets)
        parameters = [outputs]
        for w in self._weights:
            parameters.append(w)
        for b in self._biases:
            parameters.append(b)

        loss = F.cross_entropy(outputs, targets)
        for p in parameters:
            p.grad = None

        loss.backward()
        for p in parameters:
            p.data += -0.1 * p.grad

        self.loss = loss.item()

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            outputs = self.forward(inputs)
            self.backward(outputs, targets)
            if self._loss is not None:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {self._loss}")

    def predict(self, inputs):
        with torch.no_grad():
            return self.forward(inputs)
        

def aguilera_perez_rotation(v: torch.Tensor, basis: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotate a vector v around a subspace defined by a basis using the Aguilera-Pérez algorithm.

    Parameters:
    - v: torch.Tensor, the vector to be rotated (shape: [n, 1] where n is the dimension)
    - basis: torch.Tensor, the basis of the subspace (shape: [n, m] where m is the subspace dimension)
    - angle: float, the angle of rotation in radians

    Returns:
    - torch.Tensor, the rotated vector (same shape as v)
    """
    # Step 1: Normalize the basis vectors (ensure it's an orthonormal basis)
    Q, _ = torch.linalg.qr(basis)  # QR decomposition to orthonormalize basis
    m = Q.shape[1]  # Dimension of the subspace
    n = Q.shape[0]  # Dimension of the full space

    # Step 2: Project v onto the subspace
    v_projected = Q.T @ v  # Project v into the subspace (shape: [m, 1])

    # Step 3: Create a rotation matrix within the subspace
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    rotation_matrix = torch.eye(m)  # Identity matrix for the subspace
    if m >= 2:
        rotation_matrix[0, 0] = cos_theta
        rotation_matrix[0, 1] = -sin_theta
        rotation_matrix[1, 0] = sin_theta
        rotation_matrix[1, 1] = cos_theta

    # Step 4: Rotate the vector in the subspace
    v_rotated_projected = rotation_matrix @ v_projected  # Rotate within the subspace

    # Step 5: Reconstruct the full-space vector
    v_rotated = Q @ v_rotated_projected  # Map back to the original space (shape: [n, 1])

    return v_rotated


class NDVector:
    def __init__(self, n_dims=2):
        self.n_dims = n_dims
        self.dim_ids = [[i, i+n_dims] for i in range(0, n_dims)]
        self.plane_ids = []
        for axis_1_pos, axis_1_neg in self.dim_ids:
            for axis_2_pos, axis_2_neg in self.dim_ids:
                if axis_1_pos == axis_2_pos or axis_1_neg == axis_2_neg:
                    continue
                
                self.plane_ids.append([axis_1_pos, axis_2_pos, axis_1_neg, axis_2_neg])
        
        self.axis_lookup = {i: v.float() for i, v in enumerate(self.get_axis_tensors())}
        
    def get_axis_tensors(self):
        v = torch.stack([i for i in F.one_hot(torch.arange(0, self.n_dims) % self.n_dims)]).repeat(2, 1)
        v[self.n_dims:] *= -1
        return v

    def turn_left(self, dim):
        tmp = self.axis_lookup[self.plane_ids[dim][0]]
        for i in range(3):
            self.axis_lookup[self.plane_ids[dim][i]] = self.axis_lookup[self.plane_ids[dim][i+1]]
            
        self.axis_lookup[self.plane_ids[dim][-1]] = tmp
        
    def turn_right(self, dim):
        tmp = self.axis_lookup[self.plane_ids[dim][-1]]
        for i in range(3):
            self.axis_lookup[self.plane_ids[dim][3-i]] = self.axis_lookup[self.plane_ids[dim][2-i]]
            
        self.axis_lookup[self.plane_ids[dim][0]] = tmp
        
    def turn_left_angle(self, dim, theta=torch.pi / 2):
        basis_vector_x = self.axis_lookup[self.plane_ids[dim][0]]
        basis_vector_y = self.axis_lookup[self.plane_ids[dim][1]]
        basis = torch.stack([basis_vector_x, basis_vector_y])
        for i in range(4):
            old_vector = self.axis_lookup[self.plane_ids[dim][i]]
            rotated_vector = aguilera_perez_rotation(old_vector, basis.T, torch.tensor(theta))
            self.axis_lookup[self.plane_ids[dim][i]] = rotated_vector
            
    def turn_right_angle(self, dim, theta=torch.pi / 2):
        self.turn_left_angle(dim, -theta)
        
class GameObject:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.pos = torch.zeros(n_dims, dtype=torch.float32)
        self.vel = NDVector(n_dims)
        
    def facing(self, pos: torch.Tensor):
        assert pos.shape[0] == self.n_dims
        norms = torch.stack(list(self.vel.axis_lookup.values()))
        return F.cosine_similarity((self.pos - pos).repeat(self.n_dims * 2, 1), norms, dim=1)
        
    def batch_facing(self, pos: torch.Tensor):
        # TODO: Vectorize this method
        return torch.stack([self.facing(p) for p in pos])
        
    def distance_from_bounds(self, scale: int):
        norms = torch.stack(list(self.vel.axis_lookup.values()))
        return torch.sum(torch.abs(scale * norms) - torch.abs(self.pos * norms), dim=-1)
        
    def batch_distance_from_bounds(self, pos: torch.Tensor):
        return torch.cdist(self.pos.unsqueeze(0), pos.float(), p=2)

if __name__ == '__main__':
    food = torch.tensor([1, 9, 0], dtype=torch.float32)
    obj = GameObject(3)
    vec = NDVector(3)
    print(list(vec.axis_lookup.values()))
    vec.turn_left_angle(0)
    print(list(vec.axis_lookup.values()))
    vec.turn_left_angle(0)
    print(list(vec.axis_lookup.values()))