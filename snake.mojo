from python import Python
from population import dtype, neural_network_spec, game_width, game_width_offset, game_height, game_height_offset, starting_score, game_scale, ttl
from neural_network import NeuralNetwork
from math import abs, sqrt
from tensor import Tensor, TensorSpec
from logger import Logger


struct Snake(Hashable):
    var position: SIMD[dtype, 2]
    var direction: SIMD[dtype, 2]
    var score: Int
    var min_dist: SIMD[dtype, 1]
    var neural_network: NeuralNetwork[dtype]
    var history: List[SIMD[dtype, 2]]
    var fitness: Int

    fn __init__(inout self) raises:
        self.position = SIMD[dtype, 2](0, 0)
        self.direction = SIMD[dtype, 2](-1, 0)
        self.neural_network = NeuralNetwork[dtype](spec=neural_network_spec)
        self.score = starting_score
        self.min_dist = 0
        self.history = List[SIMD[dtype, 2]]()
        self.fitness = ttl
        for i in range(self.score):
            self.history.append(self.position + SIMD[dtype, 2](self.score - i - 1, 0))

    # Make a Snake instance and transfer ownership of NeuralNetwork
    fn __init__(inout self, owned neural_network: NeuralNetwork[dtype]):
        self.position = SIMD[dtype, 2](0, 0)
        self.direction = SIMD[dtype, 2](-1, 0)
        self.neural_network = neural_network
        self.score = starting_score
        self.min_dist = 0
        self.history = List[SIMD[dtype, 2]]()
        self.fitness = ttl
        for i in range(self.score):
            self.history.append(self.position + SIMD[dtype, 2](self.score - i - 1, 0))
        
    fn __moveinit__(inout self, owned existing: Self):
        self = Self(existing.neural_network)

    fn __contains__(self, point: SIMD[dtype, 2]) -> SIMD[DType.bool, 1]:
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
        self.position = SIMD[dtype, 2](0, 0)
        self.direction = SIMD[dtype, 2](-1, 0)
        self.score = starting_score
        self.min_dist = 0
        self.history = List[SIMD[dtype, 2]]()
        self.fitness = ttl
        for i in range(self.score):
            self.history.append(self.position + SIMD[dtype, 2](self.score - i - 1, 0))

    fn is_dead(self) -> Bool:
        return self.direction[0].to_int() == 0 and self.direction[1].to_int() == 0

    fn update(inout self, fruit_position: SIMD[dtype, 2], borrowed food_array_length: Int, inout screen: PythonObject, inout font: PythonObject, stats: Dict[String, Float32]) raises:
        if self.is_dead():
            return

        var torch = Python.import_module("torch")

        var fruit_left = (fruit_position < self.position)[0].to_int()
        var fruit_right = (fruit_position > self.position)[0].to_int()
        var fruit_top = (fruit_position < self.position)[1].to_int()
        var fruit_bottom = (fruit_position > self.position)[1].to_int()
        
        var wall_left = ~Snake.in_bounds(self.position + SIMD[dtype, 2](-1, 0)).to_int()
        var wall_right = ~Snake.in_bounds(self.position + SIMD[dtype, 2](1, 0)).to_int()
        var wall_top = ~Snake.in_bounds(self.position + SIMD[dtype, 2](0, -1)).to_int()
        var wall_bottom = ~Snake.in_bounds(self.position + SIMD[dtype, 2](0, 1)).to_int()

        var body_left = (self.position + SIMD[dtype, 2](-1, 0) in self).to_int()
        var body_right = (self.position + SIMD[dtype, 2](1, 0) in self).to_int()
        var body_top = (self.position + SIMD[dtype, 2](0, -1) in self).to_int()
        var body_bottom = (self.position + SIMD[dtype, 2](0, 1) in self).to_int()

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
            fruit_ahead, fruit_left_side, fruit_right_side, fruit_behind,
            wall_ahead, wall_left_side, wall_right_side
        ]).to(torch.float32).unsqueeze(1)

        var output = self.neural_network.feed(input)
        output = torch.flatten(output)
        var sorted: PythonObject
        var indices: PythonObject
        output = torch.argmax(output)

        var direction_num = 0
        if SIMD[dtype, 2](0, -1) == self.direction:
            direction_num = 0
        elif SIMD[dtype, 2](1, 0) == self.direction:
            direction_num = 1
        elif SIMD[dtype, 2](0, 1) == self.direction:
            direction_num = 2
        elif SIMD[dtype, 2](-1, 0) == self.direction:
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
            self.direction = SIMD[dtype, 2](0, -1)
        elif direction_num == 1:
            # right
            self.direction = SIMD[dtype, 2](1, 0)
        elif direction_num == 2:
            # down
            self.direction = SIMD[dtype, 2](0, 1)
        elif direction_num == 3:
            # left
            self.direction = SIMD[dtype, 2](-1, 0)

        var old_fitness = self.fitness
        self.move(fruit_position)
        self.draw(food_array_length, screen, font, old_fitness)
        #print(self.position)

    @staticmethod
    fn in_bounds(position: SIMD[dtype, 2]) -> SIMD[DType.bool, 1]:
        return position[0] >= -game_width_offset and position[0] < game_width_offset  and position[1] >= -game_height_offset and position[1] < game_height_offset 

    fn distance(self, point: SIMD[dtype, 2]) -> SIMD[dtype, 1]:
        return sqrt((self.position[0] - point[0])**2 + (self.position[1] - point[1])**2)

    @staticmethod
    fn distance(point_a: SIMD[dtype, 2], point_b: SIMD[dtype, 2]) -> SIMD[dtype, 1]:
        return sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)

    fn move(inout self, fruit_position: SIMD[dtype, 2]):
        # Death detection
        if self.position + self.direction in self or not Snake.in_bounds(self.position):
            # Active death: If the snake hits the game bounds or tail
            self.direction = SIMD[dtype, 2].splat(0)
            return
        elif self.fitness <= (self.score - starting_score) * ttl:
            # Passive death: If the snake moves too much without eating food
            self.direction = SIMD[dtype, 2].splat(0)
            return

            

        # Food detection
        if (self.position + self.direction)[0] == fruit_position[0] and (self.position + self.direction)[1] == fruit_position[1]:
            self.score += 1
            self.fitness = (self.score - starting_score + 1) * ttl - (ttl // 2)
        else:
            var current_distance = self.distance(fruit_position)
            var next_distance = Self.distance(self.position + self.direction, fruit_position)
            if next_distance < current_distance:
                self.fitness += 1
            else:
                self.fitness -= 5

        

        self.position += self.direction
        
        self.history.append(self.position)
        while len(self.history) > self.score:
            self.history = self.history[1:]

    fn generate_offspring(inout parent_a: Snake, inout parent_b: Snake) raises:
        parent_a.neural_network.average(parent_b.neural_network)
        parent_b.neural_network.average(parent_b.neural_network)

    # Draws visual representation of this Snake object to the running pygame window
    fn draw(borrowed self, current_food_count: Int, screen: PythonObject, font: PythonObject, old_fitness: Int) raises:
        var pygame = Python.import_module("pygame")
        var freetype = pygame.freetype
        # Snakes that have eaten the most food off their generation are the brightest
        var true_score = self.score - starting_score
        var main_weight = int((true_score + 1) / current_food_count * 255)
        
        # Draw the body
        var count = 0
        pygame.font.init()
        
        var color = (200, 30, 30)
        for b in self.history:
            var body_part = b[]
            var x_position = body_part[0].to_int() + game_width_offset
            var y_position = body_part[1].to_int() + game_width_offset
            var rect = (x_position * game_scale, y_position * game_scale, game_scale, game_scale)
            var rect_pointer = (x_position * game_scale, y_position * game_scale, game_scale, game_scale)
            '''if self.direction == SIMD[dtype, 2](-1, 0):
                rect_pointer = (x_position * game_scale + , y_position * game_scale, game_scale, game_scale)
            elif self.direction == SIMD[dtype, 2](1, 0):
                rect_pointer = (x_position * game_scale, y_position * game_scale, game_scale, game_scale)
            elif self.direction == SIMD[dtype, 2](0, -1):
                rect_pointer = (x_position * game_scale, y_position * game_scale, game_scale, game_scale)
            elif self.direction == SIMD[dtype, 2](0, 1):
                rect_pointer = (x_position * game_scale, y_position * game_scale, game_scale, game_scale)
            var rect = (x_position * game_scale, y_position * game_scale, game_scale, game_scale)'''
            '''var body_part_position = (self.score - 1) - count / (self.score - 1)
            var body_part_position_normalized = 255 - (255 * body_part_position)
            var fitness_deviation = self.fitness / (1 + stats["average"])
            var max_deviation = stats["max"] / (1 + stats["average"])
            var deviation_delta = max_deviation - fitness_deviation
            deviation_delta = math.clamp(deviation_delta, 0, max_deviation)
            var fitness_normalized = 255 - (255 / (1 + deviation_delta))
            var body_part_color = ((body_part_position_normalized * 0.5).to_int(), fitness_normalized.to_int(), (body_part_position_normalized * 0.5).to_int())'''
            var body_part_color = (200, 20, 20)
            if self.fitness >= old_fitness:
                body_part_color = (20, 200, 20)

            
            pygame.draw.rect(screen, body_part_color, rect)
            if count == self.score - 1:
                var text_surface = font.render(str(self.fitness), False, (255, 255, 255))
                screen.blit(text_surface, (x_position * game_scale - game_scale, y_position * game_scale))
                
            count += 1



    
    