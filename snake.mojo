from python import Python
from population import SPEC, Vector2D, VectorComponent, dtype, neural_network_spec, game_width, game_width_offset, game_height, game_height_offset, starting_score, game_scale, ttl
from neural_network import NeuralNetwork
from math import abs, sqrt, clamp
from tensor import Tensor, TensorSpec
from logger import Logger


struct Snake(Hashable):
    var position: Vector2D
    var direction: Vector2D
    var score: Int
    var min_dist: VectorComponent
    var neural_network: NeuralNetwork[SPEC]
    var history: List[Vector2D]
    var fitness: Int

    fn __init__(inout self) raises:
        self.position = Vector2D(0, 0)
        self.direction = Vector2D(-1, 0)
        self.neural_network = NeuralNetwork[SPEC]()
        self.score = starting_score
        self.min_dist = 0
        self.history = List[Vector2D]()
        self.fitness = ttl
        for i in range(self.score):
            self.history.append(self.position + Vector2D(self.score - i - 1, 0))

    # Make a Snake instance and transfer ownership of NeuralNetwork
    fn __init__(inout self, owned neural_network: NeuralNetwork[SPEC]):
        self.position = Vector2D(0, 0)
        self.direction = Vector2D(-1, 0)
        self.neural_network = neural_network
        self.score = starting_score
        self.min_dist = 0
        self.history = List[Vector2D]()
        self.fitness = ttl
        for i in range(self.score):
            self.history.append(self.position + Vector2D(self.score - i - 1, 0))
        
    fn __moveinit__(inout self, owned existing: Self):
        self = Self(existing.neural_network)

    fn __contains__(self, point: Vector2D) -> SIMD[DType.bool, 1]:
        for p in self.history:
            if p[] == point:
                return True
        return False

    fn __hash__(self) -> Int:
        return hash(self.neural_network)

    fn __str__(self) -> String:
        var result: String = "Snake("
        result += "\nPosition" + str(self.position) + ", "
        result += "\nScore" + str(self.score) + ", "
        result += "\nMinimum Distance" + str(self.min_dist) + ", "
        result += "\nHistory: ["
        for i in self.history:
            result += str(i[]) + ", "
        result += "]"
        return result

    fn __del__(owned self):
        self.neural_network^.__del__()

    fn reset(inout self):
        self.position = Vector2D(0, 0)
        self.direction = Vector2D(-1, 0)
        self.score = starting_score
        self.min_dist = 0
        self.history = List[Vector2D]()
        self.fitness = ttl
        for i in range(self.score):
            self.history.append(self.position + Vector2D(self.score - i - 1, 0))

    fn is_dead(self) -> Bool:
        return self.direction[0].to_int() == 0 and self.direction[1].to_int() == 0

    fn update(inout self, fruit_position: Vector2D, borrowed food_array_length: Int, inout screen: PythonObject, inout font: PythonObject, stats: Dict[String, Float32]) raises:
        if self.is_dead():
            return

        var torch = Python.import_module("torch")
        var pygame = Python.import_module("pygame")

        var fruit_left = (fruit_position < self.position)[0].to_int()
        var fruit_right = (fruit_position > self.position)[0].to_int()
        var fruit_top = (fruit_position < self.position)[1].to_int()
        var fruit_bottom = (fruit_position > self.position)[1].to_int()
        
        var wall_left = ~Snake.in_bounds(self.position + Vector2D(-1, 0)).to_int()
        var wall_right = ~Snake.in_bounds(self.position + Vector2D(1, 0)).to_int()
        var wall_top = ~Snake.in_bounds(self.position + Vector2D(0, -1)).to_int()
        var wall_bottom = ~Snake.in_bounds(self.position + Vector2D(0, 1)).to_int()

        var body_left = (self.position + Vector2D(-1, 0) in self).to_int()
        var body_right = (self.position + Vector2D(1, 0) in self).to_int()
        var body_top = (self.position + Vector2D(0, -1) in self).to_int()
        var body_bottom = (self.position + Vector2D(0, 1) in self).to_int()

        var facing_left = (self.direction[0] == -1).to_int()
        var facing_right = (self.direction[0] == 1).to_int()
        var facing_top = (self.direction[1] == -1).to_int()
        var facing_bottom = (self.direction[1] == 1).to_int()

        var fruit_ahead = (fruit_left and facing_left) or (fruit_right and facing_right) or (fruit_top and facing_top) or (fruit_bottom and facing_bottom)
        var fruit_left_side = (fruit_left and facing_top) or (fruit_right and facing_bottom) or (fruit_top and facing_right) or (fruit_bottom and facing_left)
        var fruit_right_side = (fruit_left and facing_bottom) or (fruit_right and facing_top) or (fruit_top and facing_left) or (fruit_bottom and facing_right)
        var fruit_behind = SIMD[DType.bool, 1]((not fruit_ahead) and (not fruit_left) and (not fruit_right)).to_int()
        var wall_ahead = Self.in_bounds(self.position + self.direction).to_int()
        var wall_left_side = (wall_left and facing_top) or (wall_right and facing_bottom) or (wall_top and facing_right) or (wall_bottom and facing_left)
        var wall_right_side = (wall_left and facing_bottom) or (wall_right and facing_top) or (wall_top and facing_left) or (wall_bottom and facing_right)
        var body_ahead = (body_left and facing_left) or (body_right and facing_right) or (body_top and facing_top) or (body_bottom and facing_bottom)
        var body_left_side = (body_left and facing_top) or (body_right and facing_bottom) or (body_top and facing_right) or (body_bottom and facing_left)
        var body_right_side = (body_left and facing_bottom) or (body_right and facing_top) or (body_top and facing_left) or (body_bottom and facing_right)

        var input = torch.tensor([
            body_ahead, body_left_side, body_right_side,
            fruit_ahead and not body_ahead, fruit_left_side and not body_left_side, fruit_right_side and not body_right_side, fruit_behind,
            wall_ahead, wall_left_side, wall_right_side
        ]).to(torch.float32).unsqueeze(1)

        var output = self.neural_network.feed(input)
        output = torch.flatten(output)
        var sorted: PythonObject
        var indices: PythonObject
        output = torch.argmax(output)

        var direction_num = 0
        if Vector2D(0, -1) == self.direction:
            direction_num = 0
        elif Vector2D(1, 0) == self.direction:
            direction_num = 1
        elif Vector2D(0, 1) == self.direction:
            direction_num = 2
        elif Vector2D(-1, 0) == self.direction:
            direction_num = 3

        if output == 0:
            # left turn
            direction_num += 1
        elif output == 1:
            # right turn
            direction_num -= 1
        
        if direction_num < 0:
            direction_num = 3
        elif direction_num > 3:
            direction_num = 0

        if direction_num == 0:
            # up
            self.direction = Vector2D(0, -1)
        elif direction_num == 1:
            # right
            self.direction = Vector2D(1, 0)
        elif direction_num == 2:
            # down
            self.direction = Vector2D(0, 1)
        elif direction_num == 3:
            # left
            self.direction = Vector2D(-1, 0)

        var old_fitness = self.fitness
        self.move(fruit_position)
        self.draw(food_array_length, screen, font, old_fitness, stats["best"].to_int())
        #print(self.position)

    @staticmethod
    fn in_bounds(position: Vector2D) -> SIMD[DType.bool, 1]:
        return position[0] >= -game_width_offset and position[0] < game_width_offset  and position[1] >= -game_height_offset and position[1] < game_height_offset 

    @staticmethod
    fn distance(point_a: Vector2D, point_b: Vector2D) -> VectorComponent:
        return sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)

    fn move(inout self, fruit_position: Vector2D):
        var next_position = self.position + self.direction
        # Death detection
        var active_death = next_position in self or not Snake.in_bounds(self.position)
        var passive_death = self.fitness <= (self.score - starting_score) * ttl
        if active_death or passive_death:
            # Active death: If the snake hits the game bounds or tail
            self.direction = Vector2D.splat(0)
            return

        # Food detection
        if next_position == fruit_position:
            self.score += 1
            self.fitness = (self.score - starting_score + 1) * ttl - (ttl // 2)
        else:
            var current_distance = Self.distance(self.position, fruit_position)
            var next_distance = Self.distance(next_position, fruit_position)
            if next_distance < current_distance:
                self.fitness += 1
            else:
                self.fitness -= 5

        self.position = next_position
        self.history.append(self.position)
        while len(self.history) > self.score:
            self.history = self.history[1:]

    # Draws visual representation of this Snake object to the running pygame window
    fn draw(borrowed self, current_food_count: Int, screen: PythonObject, font: PythonObject, old_fitness: Int, best_fitness: Int) raises:
        var pygame = Python.import_module("pygame")
        # Draw the body
        var count = 0
        var color = (200, 30, 30)
        for b in self.history:
            var body_part = b[]
            var x_position = body_part[0].to_int() + game_width_offset
            var y_position = body_part[1].to_int() + game_width_offset
            var rect = (x_position * game_scale, y_position * game_scale, game_scale, game_scale)
            var weight = clamp(self.fitness / best_fitness * 200, 0, 200).to_int()
            var body_part_color = (200 - weight, weight, 20)
            pygame.draw.rect(screen, body_part_color, rect)
            if count == self.score - 1:
                var text_surface = font.render(str(self.fitness), False, (255, 255, 255))
                screen.blit(text_surface, (x_position * game_scale - game_scale, y_position * game_scale))
                
            count += 1
