from python import Python
from population import Vec3, Vec1, DTYPE, GAME_WIDTH_OFFSET, GAME_HEIGHT_OFFSET, GAME_DEPTH_OFFSET, GAME_DEPTH, INITIAL_SCORE, GAME_SCALE, TTL, SHAPE
from neural_network import NeuralNetwork
from math import abs, sqrt, clamp
from tensor import Tensor, TensorSpec, TensorShape
from logger import Logger
from game_object import GameObject3D

# Cardinal directions in 3d: north (y+), south (y-), east (right/+x), west (left/-x), zenith (up/-z), nadir (down/+z), [4D: ana, kata]

fn move_forward(borrowed snake_position: PythonObject, inout snake_orientation: PythonObject, borrowed forward_vector: PythonObject, borrowed distance: Int) raises -> PythonObject:
    # Rotate the forward vector to align with the snake's orientation
    var rotated_forward_vector = snake_orientation.rotate(forward_vector)
    # Scale the rotated forward vector by the desired distance
    var translation_vector = rotated_forward_vector * distance
    # Update the snake's position by adding the translation vector
    var new_position = snake_position + translation_vector
    return new_position

struct Snake(Hashable):
    var game_object: GameObject3D
    var score: Int
    var min_dist: Vec1
    var neural_network: NeuralNetwork[DTYPE]
    var history: List[Vec3]
    var fitness: Vec1

    fn __init__(inout self) raises:
        self.game_object = GameObject3D(0, 0, 0)
        self.neural_network = NeuralNetwork[DTYPE]()
        self.score = INITIAL_SCORE
        self.min_dist = 0
        self.history = List[Vec3]()
        self.fitness = TTL
        var offset = Vec3(TensorShape(3), self.game_object[0], self.game_object[1], self.game_object[2])
        for i in range(self.score):
            offset[0] = self.game_object[0] + self.score - i - 1
            self.history.append(offset)

    # Make a Snake instance and transfer ownership of NeuralNetwork
    fn __init__(inout self, owned neural_network: NeuralNetwork[DTYPE]):
        self.game_object = GameObject3D(0, 0, 0)
        self.neural_network = neural_network
        self.score = INITIAL_SCORE
        self.min_dist = 0
        self.history = List[Vec3]()
        self.fitness = TTL
        var offset = Vec3(TensorShape(3), self.game_object[0], self.game_object[1], self.game_object[2])
        for i in range(self.score):
            offset[0] = self.game_object[0] + self.score - i - 1
            self.history.append(offset)
        
    fn __moveinit__(inout self, owned existing: Self):
        self = Self(existing.neural_network)

    fn __contains__(self, other: Vec3) -> SIMD[DType.bool, 1]:
        for p in self.history:
            if p[] == other:
                return True
        return False

    fn __hash__(self) -> Int:
        return hash(self.neural_network)

    '''fn __str__(self) -> String:
        var result: String = "Snake("
        result += "\nPosition" + str(self.position) + ", "
        result += "\nScore" + str(self.score) + ", "
        result += "\nMinimum Distance" + str(self.min_dist) + ", "
        result += "\nHistory: ["
        for i in self.history:
            result += str(i[]) + ", "
        result += "]"
        return result'''

    fn __del__(owned self):
        self.neural_network^.__del__()

    fn reset(inout self) raises:
        self.game_object.reset()
        self.score = INITIAL_SCORE
        self.min_dist = 0
        self.history = List[Vec3]()
        self.fitness = TTL
        var offset = Vec3(TensorShape(3), self.game_object[0], self.game_object[1], self.game_object[2])
        for i in range(self.score):
            offset[0] = self.game_object[0] + self.score - i - 1
            self.history.append(offset)

    fn is_dead(self) -> Bool:
        return False#self.direction[0].to_int() == 0 and self.direction[1].to_int() == 0 and self.direction[2].to_int() == 0

    fn update(inout self, screen: PythonObject, fruit_position: GameObject3D, stats: Dict[String, Float32]) raises:
        if self.is_dead():
            return

        var torch = Python.import_module("torch")
        var pygame = Python.import_module("pygame")

        var fruit_above = int(self.game_object.is_above_strict(fruit_position))
        var fruit_below = int(self.game_object.is_below_strict(fruit_position))
        var fruit_front = int(self.game_object.is_ahead_strict(fruit_position))
        var fruit_left = int(self.game_object.is_left_strict(fruit_position))
        var fruit_right = int(self.game_object.is_right_strict(fruit_position))

        var wall_above = int(Self.in_bounds(self.game_object.position + self.game_object["up"]))
        var wall_below = int(Self.in_bounds(self.game_object.position + self.game_object["down"]))
        var wall_front = int(Self.in_bounds(self.game_object.position + self.game_object["front"]))
        var wall_left = int(Self.in_bounds(self.game_object.position + self.game_object["left"]))
        var wall_right = int(Self.in_bounds(self.game_object.position + self.game_object["right"]))

        var body_above = int(self.game_object.position + self.game_object["up"] in self)
        var body_below = int(self.game_object.position + self.game_object["down"] in self)
        var body_front = int(self.game_object.position + self.game_object["front"] in self)
        var body_left = int(self.game_object.position + self.game_object["left"] in self)
        var body_right = int(self.game_object.position + self.game_object["right"] in self)

        var input = torch.tensor([
            fruit_above, fruit_below, fruit_front, fruit_left, fruit_right,
            wall_above, wall_below, wall_front, wall_left, wall_right,
            body_above, body_below, body_front, body_left, body_right
        ]).to(torch.float32).unsqueeze(1)

        var output = self.neural_network.feed(input)
        output = torch.flatten(output)
        var sorted: PythonObject
        var indices: PythonObject
        output = torch.argmax(output)

        if output == 0:
            self.game_object.turn_left()
        elif output == 1:
            self.game_object.turn_right()
        elif output == 2:
            self.game_object.turn_up()
        elif output == 3:
            self.game_object.turn_down()
        '''elif output == 5:
            pass'''

        var old_fitness = self.fitness
        self.move(fruit_position)
        self.draw(screen, old_fitness, stats["best_fitness"])
        #print(self.position)

    @staticmethod
    fn in_bounds(position: Vec3) -> SIMD[DType.bool, 1]:
        return position[0] >= -GAME_WIDTH_OFFSET and position[0] < GAME_WIDTH_OFFSET  and position[1] >= -GAME_HEIGHT_OFFSET and position[1] < GAME_HEIGHT_OFFSET and position[2] >= -GAME_DEPTH_OFFSET and position[2] < GAME_DEPTH_OFFSET

    @staticmethod
    fn distance(point_a: Vec3, point_b: Vec3) -> Vec1:
        return sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2 + (point_a[2] - point_b[2])**2)

    fn move(inout self, fruit_position: GameObject3D) raises:
        var next_position = self.game_object.position + self.game_object["front"]
        # Death detection
        var active_death = next_position in self or not Snake.in_bounds(self.game_object.position)
        var passive_death = self.fitness <= (self.score - INITIAL_SCORE) * TTL
        if active_death or passive_death:
            # Active death: If the snake hits the game bounds or tail
            #self.direction = Vec3.splat(0)
            return

        # Food detection
        if next_position == fruit_position.position:
            self.score += 1
            self.fitness = (self.score - INITIAL_SCORE + 1) * TTL# + ((self.score - INITIAL_SCORE) * (TTL // 8))
        else:
            var current_distance = fruit_position.distance(self.game_object)
            var next_distance = fruit_position.distance(next_position)
            if next_distance < current_distance:
                self.fitness += 1
            else:
                self.fitness -= 5

        self.game_object.move_forward()
        self.history.append(self.game_object.position)
        while len(self.history) > self.score:
            self.history = self.history[1:]

    # Draws visual representation of this Snake object to the running pygame window
    fn draw(borrowed self, screen: PythonObject, old_fitness: Vec1, best_fitness: Vec1) raises:
        var pygame = Python.import_module("pygame")
        var font = pygame.font.SysFont('Comic Sans MS', 20)
        # Draw the body
        var count = 0
        var color = (200, 30, 30)
        for b in self.history:
            var body_part = b[]
            var x_position = int(body_part[0]) + GAME_WIDTH_OFFSET
            var y_position = int(body_part[1]) + GAME_WIDTH_OFFSET
            var z_position = int(body_part[2]) + GAME_DEPTH_OFFSET
            var z_value = clamp(z_position / GAME_DEPTH, 0, 1)
            var rect = (x_position * GAME_SCALE, y_position * GAME_SCALE, GAME_SCALE, GAME_SCALE)
            var weight = int(clamp(self.fitness / best_fitness * 200, 0, 200))
            var weight_z = int(clamp(z_value * 200, 0, 200))
            var body_part_color = (200 - weight_z, weight, 20)
            pygame.draw.rect(screen, body_part_color, rect)
            if count == self.score - 1:
                var text_surface = font.render(str(self.fitness), False, (255, 255, 255))
                screen.blit(text_surface, (x_position * GAME_SCALE - GAME_SCALE, y_position * GAME_SCALE))
                
            count += 1