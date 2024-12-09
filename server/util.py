import torch
import torch.nn as nn

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
        self.dim_ids = [[i, i+1] for i in range(0, n_dims * 2, 2)]
        self.plane_ids = []
        for axis_1_pos, axis_1_neg in self.dim_ids:
            for axis_2_pos, axis_2_neg in self.dim_ids:
                if axis_1_pos == axis_2_pos or axis_1_neg == axis_2_neg:
                    continue
                
                self.plane_ids.append([axis_1_pos, axis_1_neg, axis_2_pos, axis_2_neg])
        self.axis_lookup = {i: i for i in range(n_dims*2)}

    def turn_left(self, dim):
        tmp = self.plane_ids[dim][0]
        for i in range(3):
            print(self.plane_ids[dim][i], self.plane_ids[dim][i+1])
            self.axis_lookup[self.plane_ids[dim][i]] = self.axis_lookup[self.plane_ids[dim][i+1]]
            
        self.axis_lookup[self.plane_ids[dim][-1]] = tmp
        
    def turn_right(self, dim):
        tmp = self.plane_ids[dim][-1]
        for i in range(3):
            self.axis_lookup[self.plane_ids[dim][i+1]] = self.axis_lookup[self.plane_ids[dim][i]]
            
        self.axis_lookup[self.plane_ids[dim][0]] = tmp
        
if __name__ == '__main__':
    v = Vector(4)
    print(v.axis_lookup)
    print(v.plane_ids)
    v.turn_right(0)
    print(v.axis_lookup)