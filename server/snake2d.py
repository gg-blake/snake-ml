import torch
from util import NeuralNetwork

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

    def step(self, key: str | int, food: torch.Tensor):
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

    def get_inputs(self, food_pos: torch.Tensor):
        food_dir = food_pos - self.pos
        food_above = food_dir[1] > 0
        food_below = food_pos[1] < 0
        food_west = food_dir[0] < 0
        food_east = food_pos[0] > 0
        
        food_ahead = self.vel[0] == 1 and food_east or self.vel[0] == -1 and food_west or self.vel[1] == 1 and food_above or self.vel[1] == -1 and food_below
        food_left = self.vel[0] == 1 and food_above or self.vel[0] == -1 and food_below or self.vel[1] == 1 and food_west or self.vel[1] == -1 and food_east
        food_right = self.vel[0] == 1 and food_above or self.vel[0] == -1 and food_below or self.vel[1] == 1 and food_east or self.vel[1] == -1 and food_west
        obstacle_above = (self.pos + UP)[1] > GAME_BOUNDS or self.pos + UP in self
        obstacle_below = (self.pos + DOWN)[1] < -GAME_BOUNDS or self.pos + DOWN in self
        obstacle_west = (self.pos + LEFT)[0] < -GAME_BOUNDS or self.pos + LEFT in self
        obstacle_east = (self.pos + RIGHT)[0] > GAME_BOUNDS or self.pos + RIGHT in self
        #obstacle_ahead = self.relative_facing(torch.tensor([obstacle_east, obstacle_west, obstacle_above, obstacle_below])).any().item()
        #obstacle_left = self.relative_facing(torch.tensor([obstacle_above, obstacle_below, obstacle_west, obstacle_east])).any().item()
        #obstacle_right = self.relative_facing(torch.tensor([obstacle_above, obstacle_below, obstacle_east, obstacle_west])).any().item()
        obstacle_ahead = self.vel[0] == 1 and obstacle_east or self.vel[0] == -1 and obstacle_west or self.vel[1] == 1 and obstacle_above or self.vel[1] == -1 and obstacle_below
        obstacle_left = self.vel[0] == 1 and obstacle_above or self.vel[0] == -1 and obstacle_below or self.vel[1] == 1 and obstacle_west or self.vel[1] == -1 and obstacle_east
        obstacle_right = self.vel[0] == 1 and obstacle_above or self.vel[0] == -1 and obstacle_below or self.vel[1] == 1 and obstacle_east or self.vel[1] == -1 and obstacle_west
        food_ahead = float(food_ahead)
        food_left = float(food_left)
        food_right = float(food_right)
        obstacle_ahead = float(obstacle_ahead)
        obstacle_left = float(obstacle_left)
        obstacle_right = float(obstacle_right)

        return torch.tensor([food_ahead, food_left, food_right, obstacle_ahead, obstacle_left, obstacle_right], dtype=torch.float32)