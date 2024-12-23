import torch
from util import NeuralNetwork, GameObject
import torch.nn.functional as F

INITIAL_SCORE = 5

class NDSnake:
    def __init__(self, n_dims, hidden_size, game_bounds, speed, ttl, turn_angle=None):
        global mutation_power
        global dot
        self.n_dims = n_dims
        self.setup(game_bounds, turn_angle, speed, ttl)
        self.input_size = self.n_dims * 6 - 2
        # Turn left and right on each axis of rotation plus and extra node for no turning
        self.output_size = len(self.game_object.vel.plane_ids)
        self.nn = NeuralNetwork(self.input_size, hidden_size, self.output_size)
        
    def __dict__(self, uid: int):
        return {
            "uid": uid,
            "pos": self.game_object.pos.tolist(),
            "vel": {i: v.tolist() for i, v in enumerate(self.game_object.vel.normals)},
            "score": self.score,
            "alive": self.alive,
            "history": [s.tolist() for s in self.history],
            "fitness": self.fitness,

        }

    def setup(self, game_bounds, turn_angle, speed, ttl):
        self.game_object = GameObject(self.n_dims)
        self.turn_angle = turn_angle
        self.speed = speed
        self.score = INITIAL_SCORE
        self.game_bounds = game_bounds
        self.ttl = ttl
        self.alive = True
        self.history = []
        for i in range(INITIAL_SCORE):
            self.history.append(torch.tensor([-self.score + i + 1, *[0 for x in range(self.n_dims - 1)]], dtype=torch.float32))
        self.fitness = self.ttl

    def step(self, logits: torch.Tensor, food: torch.Tensor, discrete=False, discrete_turn_threshold=0.1):
        if not self.alive:
            return
        
        if not discrete:
            self.game_object.vel.rotate(logits)
        else:
            if self.n_dims > 2:
                if logits[torch.argmax(torch.abs(logits))] >= 0:
                    self.game_object.vel.rotate_discrete_left(torch.argmax(torch.abs(logits)).item())
                else:
                    self.game_object.vel.rotate_discrete_right(torch.argmax(torch.abs(logits)).item())
            else:
                matrix = torch.tensor([[0, -logits[0].item()], [logits[0].item(), 0]]).unsqueeze(0).repeat(2, 1, 1)
                self.game_object.vel.normals = F.normalize(torch.bmm(matrix, self.game_object.vel.normals.unsqueeze(1).transpose(1,2)).squeeze(-1))

        

        next_position = self.game_object.pos + (self.game_object.vel.normals[0] * self.speed)
        body_parts = torch.stack(self.history[:-1])
        body_collision_death = bool(torch.sum(torch.all(next_position.repeat(self.score - 1, 1) == body_parts, dim=1)).item())
        body_collision_death = torch.any((body_parts - self.game_object.pos).pow(2).sum(-1).sqrt()[:-2] < 1)
        out_of_bounds_death = bool(torch.any(self.game_object.distance_from_bounds(self.game_bounds) == 0).item())
        active_death = body_collision_death or out_of_bounds_death
        passive_death = self.fitness <= (self.score - INITIAL_SCORE) * self.ttl
        if active_death or passive_death:
            self.alive = False
            return
            
        if (next_position - food).pow(2).sum(-1).sqrt() < 3:
            self.score += 1
            self.fitness = self.ttl + 2**(self.score - INITIAL_SCORE + 1) * self.ttl
        else:
            dist_now = (self.game_object.pos - food).pow(2).sum(-1).sqrt()
            dist_next = (next_position - food).pow(2).sum(-1).sqrt()
            if dist_next < dist_now:
                self.fitness += 1
            else:
                self.fitness -= 5
        
        self.game_object.pos = next_position
        self.history.append(self.game_object.pos)
        # Keep history length fixed
        while len(self.history) > self.score:
            self.history = self.history[1:]
            
            
    def __contains__(self, other: torch.Tensor) -> bool:
        for p in self.history:
            if int(p[0].item()) == int(other[0].item()) and int(p[1].item()) == int(other[1].item()):
                return True
        return False

    def get_inputs(self, food_pos: torch.Tensor):
        norms = (self.game_object.vel.normals * self.game_object.pos).repeat(2, 1)
        norms[self.n_dims:] *= -1
        dist = torch.cdist(norms, torch.stack([b for b in self.history[:-1]]))
        nearby_body = torch.any(torch.cat([dist[:self.n_dims], dist[self.n_dims+1:]]) == 0, dim=1).float()
        nearby_food = torch.clamp(self.game_object.facing(food_pos), min=-1, max=1)
        nearby_bounds = (self.game_object.distance_from_bounds(self.game_bounds - 1) == 0).float()
        nearby_bounds = torch.cat([nearby_bounds[:self.n_dims], nearby_bounds[self.n_dims+1:]])
        return torch.cat([nearby_body, nearby_food, nearby_bounds])

    
