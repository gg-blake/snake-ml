import torch
import torch.nn.functional as F
import asyncio
import signal
from websockets.asyncio.server import serve
import json
from time import time


GAME_BOUNDS = 10

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

class NeuralNetwork:
    def __init__(self, *layers):
        self._layers = layers
        self._weights = [torch.randn(layer.input_size, layer.output_size) for layer in layers]
        self._biases = [torch.randn(1, layer.output_size) for layer in layers]
        self._loss = None

    def forward(self, inputs):
        # Implement forward pass logic
        # Input -> Hidden Layer

        for i, layer in enumerate(self._layers):
            inputs = inputs @ self._weights[i] + self._biases[i]
            inputs = torch.relu(inputs)  # Activation function
            inputs = torch.softmax(inputs, dim=1) # Normalization function

        return inputs

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

class Vec2:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vec2(self.x / scalar, self.y / scalar)

    def __repr__(self):
        return f"Vec2({self.x}, {self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class Game:
    def __init__(self):
        self.nn = None
        self.pos = Vec2(0, 0)
        self.vel = Vec2(1, 0)
        self.dir = 0
        self.score = 5
        self.alive = True
        self.history = []
        self.data = []

    def reset(self):
        self.pos = Vec2(0, 0)
        self.vel = Vec2(1, 0)

    def step(self, key):
        if key == "w" and self.vel != Vec2(0, -1):
            self.vel = Vec2(0, 1)
        elif key == "s" and self.vel != Vec2(0, 1):
            self.vel = Vec2(0, -1)
        elif key == "a" and self.vel != Vec2(1, 0):
            self.vel = Vec2(-1, 0)
        elif key == "d" and self.vel != Vec2(-1, 0):
            self.vel = Vec2(1, 0)

        self.pos += self.vel
        self.history.append([self.pos.x, self.pos.y])
        if abs(self.pos.x) > GAME_BOUNDS or abs(self.pos.y) > GAME_BOUNDS:
            self.alive = False
        elif [self.pos.x, self.pos.y] in self.history[:-1]:
            self.alive = False

        # Keep history length fixed
        while len(self.history) > self.score:
            self.history.pop(0)

    def get_inputs(self, food_pos):
        food_above = food_pos.y > self.pos.y
        food_below = food_pos.y < self.pos.y
        food_west = food_pos.x < self.pos.x
        food_east = food_pos.x > self.pos.x
        food_ahead = self.vel == Vec2(1, 0) and food_east or self.vel == Vec2(-1, 0) and food_west or self.vel == Vec2(0, 1) and food_above or self.vel == Vec2(0, -1) and food_below
        food_left = self.vel == Vec2(1, 0) and food_above or self.vel == Vec2(-1, 0) and food_below or self.vel == Vec2(0, 1) and food_west or self.vel == Vec2(0, -1) and food_east
        food_right = self.vel == Vec2(1, 0) and food_below or self.vel == Vec2(-1, 0) and food_above or self.vel == Vec2(0, 1) and food_east or self.vel == Vec2(0, -1) and food_west
        obstacle_above = self.pos.y + 1 > GAME_BOUNDS or Vec2(self.pos.x, self.pos.y + 1) in self.history
        obstacle_below = self.pos.y - 1 < -GAME_BOUNDS or Vec2(self.pos.x, self.pos.y - 1) in self.history
        obstacle_west = self.pos.x - 1 < -GAME_BOUNDS or Vec2(self.pos.x - 1, self.pos.y) in self.history
        obstacle_east = self.pos.x + 1 > GAME_BOUNDS or Vec2(self.pos.x + 1, self.pos.y) in self.history
        obstacle_ahead = self.vel == Vec2(1, 0) and obstacle_east or self.vel == Vec2(-1, 0) and obstacle_west or self.vel == Vec2(0, 1) and obstacle_above or self.vel == Vec2(0, -1) and obstacle_below
        obstacle_left = self.vel == Vec2(1, 0) and obstacle_above or self.vel == Vec2(-1, 0) and obstacle_below or self.vel == Vec2(0, 1) and obstacle_west or self.vel == Vec2(0, -1) and obstacle_east
        obstacle_right = self.vel == Vec2(1, 0) and obstacle_below or self.vel == Vec2(-1, 0) and obstacle_above or self.vel == Vec2(0, 1) and obstacle_east or self.vel == Vec2(0, -1) and obstacle_west
        food_ahead = float(food_ahead)
        food_left = float(food_left)
        food_right = float(food_right)
        obstacle_ahead = float(obstacle_ahead)
        obstacle_left = float(obstacle_left)
        obstacle_right = float(obstacle_right)

        return torch.tensor([[food_ahead, food_left, food_right, obstacle_ahead, obstacle_left, obstacle_right]], dtype=torch.float32)

g = Game()
        
async def handle_control_channel(websocket):
    async for message in websocket:
        print(f"Received message (control): {message}")
        decoded = json.loads(message)
        g.step(decoded["key_pressed"])
        decoded["pos"] = [g.pos.x, g.pos.y]
        decoded["timestamp"] = time() / 1000
        decoded["history"] = g.history
        await websocket.send(json.dumps(decoded))


async def handle_data_channel(websocket):
    async for message in websocket:
        await websocket.send(message)
        


async def server():
    # Set the stop condition when receiving SIGTERM.
    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    loop.add_signal_handler(signal.SIGTERM, stop.set_result, None)

    async with serve(handle_control_channel, "localhost", 6600):
        await stop




if __name__ == "__main__":
    num = 0
    asyncio.run(server())
