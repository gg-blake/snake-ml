import torch
from util import NeuralNetwork, GameObject

GAME_BOUNDS = 10
INITIAL_SCORE = 5
TTL = 100
UP, DOWN, LEFT, RIGHT = torch.tensor([0, 1]), torch.tensor([0, -1]), torch.tensor([-1, 0]), torch.tensor([1, 0])

class Snake:
    def __init__(self, input_size, hidden_size, output_size):
        global mutation_power
        global dot
        self.setup()
        self.nn = NeuralNetwork(input_size, hidden_size, output_size)
        
    def __dict__(self):
        return {
            "pos": self.pos.tolist(),
            "vel": self.vel.tolist(),
            "dir": self.dir,
            "score": self.score,
            "alive": self.alive,
            "history": [s.tolist() for s in self.history],
            "fitness": self.fitness
        }

    def setup(self):
        self.pos = torch.tensor([0, 0])
        self.vel = torch.tensor([1, 0])
        self.dir = 2
        self.score = 5
        self.alive = True
        self.history = []
        for i in range(5):
            self.history.append(torch.tensor([-self.score + i + 1, 0]))
        self.fitness = TTL

    def step(self, key: str | int, food: GameObject):
        if not self.alive:
            return
        
        if isinstance(key, int):
            # -1 is counter-clockwise and +1 is clockwise
            self.dir += key
            if self.dir > 3:
                self.dir = 0
            elif self.dir < 0:
                self.dir = 3

        
        if (key == "w" and torch.eq(self.vel, DOWN).all().item()) or self.dir == 1:
            self.vel = UP
        elif (key == "s" and torch.eq(self.vel, UP).all().item()) or self.dir == 3:
            self.vel = DOWN
        elif (key == "a" and torch.eq(self.vel, RIGHT).all().item()) or self.dir == 0:
            self.vel = LEFT
        elif (key == "d" and torch.eq(self.vel, LEFT).all().item()) or self.dir == 2:
            self.vel = RIGHT
            
        next_position = self.history[-1] + self.vel
        active_death = next_position in self or abs(next_position[0]) > GAME_BOUNDS or abs(next_position[1]) > GAME_BOUNDS
        passive_death = self.fitness <= (self.score - INITIAL_SCORE) * TTL
        if active_death or passive_death:
            self.alive = False
            return
            
        if (next_position == food).all():
            self.score += 1
            self.fitness = TTL + (self.score - INITIAL_SCORE + 1) * TTL
        else:
            dist_now = (self.pos - food).pow(2).sum(-1).sqrt()
            dist_next = (next_position - food).pow(2).sum(-1).sqrt()
            if dist_next < dist_now:
                self.fitness += 1
            else:
                self.fitness -= 10

        self.pos = next_position
        self.history.append(self.pos)
        # Keep history length fixed
        while len(self.history) > self.score:
            self.history = self.history[1:]
            
            
    def __contains__(self, other: torch.Tensor) -> bool:
        for p in self.history:
            if int(p[0].item()) == int(other[0].item()) and int(p[1].item()) == int(other[1].item()):
                return True
        return False
    
    def relative_facing(self, direction_bools: torch.Tensor):
        identity = torch.stack([RIGHT, LEFT, UP, DOWN])
        direction_matrix = (identity == self.vel).all(dim=1)
        return direction_matrix & direction_bools

    def get_inputs(self, food_pos: GameObject):
        

        return torch.tensor([food_ahead, food_left, food_right, obstacle_ahead, obstacle_left, obstacle_right], dtype=torch.float32)