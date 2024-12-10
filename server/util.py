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
        


class Vector:
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
        
class GameObject:
    def __init__(self, n_dims):
        self.n_dims = n_dims
        self.pos = torch.zeros(n_dims)
        self.vel = Vector(n_dims)
        
    def facing(self, pos: torch.Tensor):
        assert pos.shape[0] == self.n_dims
        norms = torch.stack(list(self.vel.axis_lookup.values()))
        print(norms.shape, pos.shape)
        return F.cosine_similarity(norms, pos, dim=1)
        
    def batch_facing(self, pos: torch.Tensor):
        # TODO: Vectorize this method
        return torch.stack([self.facing(p) for p in pos])
        
    def distance_from_bounds(self, scale: int):
        norms = torch.stack(list(self.vel.axis_lookup.values()))
        return torch.sum(torch.abs(scale * norms) - torch.abs(self.pos * norms), dim=1)
        
if __name__ == '__main__':
    food = torch.tensor([1, 9, 0], dtype=torch.float32)
    obj = GameObject(3)
    print(obj.facing(food))
    print(obj.batch_facing(torch.stack([food, food, food])))