from python import Python
from population import Vector3D, Vector1D, DTYPE, GAME_WIDTH_OFFSET, GAME_HEIGHT_OFFSET, GAME_DEPTH_OFFSET, GAME_DEPTH, INITIAL_SCORE, GAME_SCALE, TTL, SPEC
from neural_network import NeuralNetwork
from math import abs, sqrt, clamp
from tensor import Tensor, TensorSpec
from logger import Logger

# Cardinal directions in 3d: north (y+), south (y-), east (right/+x), west (left/-x), zenith (up/-z), nadir (down/+z), [4D: ana, kata]

struct Snake(Hashable):
    var position: Vector3D
    var direction: Vector3D
    var score: Int
    var min_dist: Vector1D
    var neural_network: NeuralNetwork[SPEC]
    var history: List[Vector3D]
    var fitness: Vector1D

    fn __init__(inout self) raises:
        self.position = Vector3D(0, 0, 0)
        self.direction = Vector3D(-1, 0, 0) # Starts facing left
        self.neural_network = NeuralNetwork[SPEC]()
        self.score = INITIAL_SCORE
        self.min_dist = 0
        self.history = List[Vector3D]()
        self.fitness = TTL
        for i in range(self.score):
            self.history.append(self.position + Vector3D(self.score - i - 1, 0, 0))

    # Make a Snake instance and transfer ownership of NeuralNetwork
    fn __init__(inout self, owned neural_network: NeuralNetwork[SPEC]):
        self.position = Vector3D(0, 0 , 0)
        self.direction = Vector3D(-1, 0, 0)
        self.neural_network = neural_network
        self.score = INITIAL_SCORE
        self.min_dist = 0
        self.history = List[Vector3D]()
        self.fitness = TTL
        for i in range(self.score):
            self.history.append(self.position + Vector3D(self.score - i - 1, 0, 0))
        
    fn __moveinit__(inout self, owned existing: Self):
        self = Self(existing.neural_network)

    fn __contains__(self, point: Vector3D) -> SIMD[DType.bool, 1]:
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
        self.position = Vector3D(0, 0, 0)
        self.direction = Vector3D(-1, 0, 0)
        self.score = INITIAL_SCORE
        self.min_dist = 0
        self.history = List[Vector3D]()
        self.fitness = TTL
        for i in range(self.score):
            self.history.append(self.position + Vector3D(self.score - i - 1, 0, 0))

    fn is_dead(self) -> Bool:
        return self.direction[0].to_int() == 0 and self.direction[1].to_int() == 0 and self.direction[2].to_int() == 0

    fn update(inout self, screen: PythonObject, fruit_position: Vector3D, stats: Dict[String, Float32]) raises:
        if self.is_dead():
            return

        var torch = Python.import_module("torch")
        var pygame = Python.import_module("pygame")

        # Absolute directions for fruit
        var fruit_west = (fruit_position < self.position)[0].to_int()
        var fruit_east = (fruit_position > self.position)[0].to_int()
        var fruit_north = (fruit_position < self.position)[1].to_int()
        var fruit_south = (fruit_position > self.position)[1].to_int()
        var fruit_zenith = (fruit_position < self.position)[2].to_int()
        var fruit_nadir = (fruit_position > self.position)[2].to_int()
        
        # Absolute directions for walls
        var wall_west = ~Snake.in_bounds(self.position + Vector3D(-1, 0, 0)).to_int()
        var wall_east = ~Snake.in_bounds(self.position + Vector3D(1, 0, 0)).to_int()
        var wall_north = ~Snake.in_bounds(self.position + Vector3D(0, -1, 0)).to_int()
        var wall_south = ~Snake.in_bounds(self.position + Vector3D(0, 1, 0)).to_int()
        var wall_zenith = ~Snake.in_bounds(self.position + Vector3D(0, 0, -1)).to_int()
        var wall_nadir = ~Snake.in_bounds(self.position + Vector3D(0, 0, 1)).to_int()

        # Absolute directions for body
        var body_west = (self.position + Vector3D(-1, 0, 0) in self).to_int()
        var body_east = (self.position + Vector3D(1, 0, 0) in self).to_int()
        var body_north = (self.position + Vector3D(0, -1, 0) in self).to_int()
        var body_south = (self.position + Vector3D(0, 1, 0) in self).to_int()
        var body_zenith = (self.position + Vector3D(0, 0, -1) in self).to_int()
        var body_nadir = (self.position + Vector3D(0, 0, 1) in self).to_int()

        # Absolute directions
        var facing_west = (self.direction[0] == -1).to_int()
        var facing_east = (self.direction[0] == 1).to_int()
        var facing_north = (self.direction[1] == -1).to_int()
        var facing_south = (self.direction[1] == 1).to_int()
        var facing_zenith = (self.direction[2] == -1).to_int()
        var facing_nadir = (self.direction[2] == 1).to_int()

        # Relative directions for fruit
        var relative_fruit_front = (fruit_north and facing_north) or (fruit_south and facing_south) or (fruit_east and facing_east) or (fruit_west and facing_west) or (fruit_zenith and facing_zenith) or (fruit_nadir and facing_nadir)
        var relative_fruit_left = (fruit_north and facing_west) or (fruit_south and facing_east) or (fruit_east and facing_north) or (fruit_west and facing_south) or (fruit_west and facing_zenith) or (fruit_east and facing_nadir)
        var relative_fruit_right = (fruit_north and facing_east) or (fruit_south and facing_west) or (fruit_east and facing_south) or (fruit_west and facing_north) or (fruit_east and facing_zenith) or (fruit_west and facing_nadir)
        var relative_fruit_above = (fruit_north and facing_zenith) or (fruit_south and facing_nadir) or (fruit_zenith and facing_south) or (fruit_nadir and facing_north) or (fruit_east and facing_zenith) or (fruit_west and facing_nadir)
        var relative_fruit_below = (fruit_north and facing_nadir) or (fruit_south and facing_zenith) or (fruit_zenith and facing_north) or (fruit_nadir and facing_south) or (fruit_east and facing_nadir) or (fruit_west and facing_zenith)
        var relative_fruit_behind = SIMD[DType.bool]((not relative_fruit_front) and (not relative_fruit_left) and (not relative_fruit_right) and (not relative_fruit_above) and (not relative_fruit_below)).to_int()
        

        # Relative directions for walls
        var relative_wall_front = int(Self.in_bounds(self.position + self.direction))
        var relative_wall_left = (wall_north and facing_west) or (wall_south and facing_east) or (wall_east and facing_north) or (wall_west and facing_south) or (wall_west and facing_zenith) or (wall_east and facing_nadir)
        var relative_wall_right = (wall_north and facing_east) or (wall_south and facing_west) or (wall_east and facing_south) or (wall_west and facing_north) or (wall_east and facing_zenith) or (wall_west and facing_nadir)
        var relative_wall_above = (wall_north and facing_zenith) or (wall_south and facing_nadir) or (wall_zenith and facing_south) or (wall_nadir and facing_north) or (wall_east and facing_zenith) or (wall_west and facing_nadir)
        var relative_wall_below = (wall_north and facing_nadir) or (wall_south and facing_zenith) or (wall_zenith and facing_north) or (wall_nadir and facing_south) or (wall_east and facing_nadir) or (wall_west and facing_zenith)
        
        # Relative directions for body
        var relative_body_front = int((self.position + self.direction) in self)
        var relative_body_left = (body_north and facing_west) or (body_south and facing_east) or (body_east and facing_north) or (body_west and facing_south) or (body_west and facing_zenith) or (body_east and facing_nadir)
        var relative_body_right = (body_north and facing_east) or (body_south and facing_west) or (body_east and facing_south) or (body_west and facing_north) or (body_east and facing_zenith) or (body_west and facing_nadir)
        var relative_body_above = (body_north and facing_zenith) or (body_south and facing_nadir) or (body_zenith and facing_south) or (body_nadir and facing_north) or (body_east and facing_zenith) or (body_west and facing_nadir)
        var relative_body_below = (body_north and facing_nadir) or (body_south and facing_zenith) or (body_zenith and facing_north) or (body_nadir and facing_south) or (body_east and facing_nadir) or (body_west and facing_zenith)
        
        # Body and fruit relative comparisons
        relative_fruit_front = ~relative_body_front and relative_fruit_front
        relative_fruit_left = ~relative_body_left and relative_fruit_left
        relative_fruit_right = ~relative_body_right and relative_fruit_right
        relative_fruit_above = ~relative_body_above and relative_fruit_above
        relative_fruit_below = ~relative_body_below and relative_fruit_below
        

        var input = torch.tensor([
            relative_fruit_front, relative_fruit_left, relative_fruit_right, relative_fruit_above, relative_fruit_below,
            relative_wall_front, relative_wall_left, relative_wall_right, relative_wall_above, relative_wall_below,
            relative_body_front, relative_body_left, relative_body_right, relative_body_above, relative_body_below
        ]).to(torch.float32).unsqueeze(1)

        var output = self.neural_network.feed(input)
        output = torch.flatten(output)
        var sorted: PythonObject
        var indices: PythonObject
        output = torch.argmax(output)

        var direction_xy = 0
        var direction_xz = 0
        var direction_yz = 0

        var direction_map_xy = List[Int](0, 1, 2, 3)
        var direction_map_xz = List[Int](0, 5, 2, 4)
        var direction_map_yz = List[Int](1, 5, 3, 4)

        var current_direction = 0

        fn index(l: List[Int], v: Int) -> Int:
            for i in range(len(l)):
                if l[i] == v:
                    return i
            return -1

        if self.direction == Vector3D(-1, 0, 0):
            direction_xy = 0
            direction_xz = 0
            direction_yz = -1
        elif self.direction == Vector3D(1, 0, 0):
            direction_xy = 2
            direction_xz = 2
            direction_yz = -1
        elif self.direction == Vector3D(0, -1, 0):
            direction_xy = 3
            direction_xz = -1
            direction_yz = 2
        elif self.direction == Vector3D(0, 1, 0):
            direction_xy = 1
            direction_xz = -1
            direction_yz = 0
        elif self.direction == Vector3D(0, 0, -1):
            direction_xy = -1
            direction_xz = 3
            direction_yz = 3
        elif self.direction == Vector3D(0, 0, 1):
            direction_xy = -1
            direction_xz = 1
            direction_yz = 1

        var left_right_index = 0
        var left_right_list = List[Int](0, 0, 0, 0)
        var up_down_index = 0
        var up_down_list = List[Int](0, 0, 0, 0)

        if direction_xy == -1:
            left_right_index = direction_xz
            left_right_list = direction_map_xz
            up_down_index = direction_yz
            up_down_list = direction_map_yz
        elif direction_xz == -1:
            left_right_index = direction_xy
            left_right_list = direction_map_xy
            up_down_index = direction_yz
            up_down_list = direction_map_yz
        elif direction_yz == -1:
            left_right_index = direction_xy
            left_right_list = direction_map_xy
            up_down_index = direction_xz
            up_down_list = direction_map_xz

        if output == 0:
            left_right_index += 1
            if left_right_index == 4:
                left_right_index = 0
            current_direction = left_right_list[left_right_index]
        elif output == 1:
            left_right_index -= 1
            if left_right_index == -1:
                left_right_index = 3
            current_direction = left_right_list[left_right_index]
        elif output == 2:
            up_down_index += 1
            if up_down_index == 4:
                up_down_index = 0
            current_direction = up_down_list[up_down_index]
        elif output == 3:
            up_down_index -= 1
            if up_down_index == -1:
                up_down_index = 3
            current_direction = up_down_list[up_down_index]
        else:
            print("Invalid output from neural network")
            return

        if current_direction == 0:
            self.direction = Vector3D(-1, 0, 0)
        elif current_direction == 1:
            self.direction = Vector3D(0, 1, 0)
        elif current_direction == 2:
            self.direction = Vector3D(1, 0, 0)
        elif current_direction == 3:
            self.direction = Vector3D(0, -1, 0)
        elif current_direction == 4:
            self.direction = Vector3D(0, 0, -1)
        elif current_direction == 5:
            self.direction = Vector3D(0, 0, 1)
        

        var old_fitness = self.fitness
        self.move(fruit_position)
        self.draw(screen, old_fitness, stats["best_fitness"])
        #print(self.position)

    @staticmethod
    fn in_bounds(position: Vector3D) -> SIMD[DType.bool, 1]:
        return position[0] >= -GAME_WIDTH_OFFSET and position[0] < GAME_WIDTH_OFFSET  and position[1] >= -GAME_HEIGHT_OFFSET and position[1] < GAME_HEIGHT_OFFSET and position[2] >= -GAME_DEPTH_OFFSET and position[2] < GAME_DEPTH_OFFSET

    @staticmethod
    fn distance(point_a: Vector3D, point_b: Vector3D) -> Vector1D:
        return sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)

    fn move(inout self, fruit_position: Vector3D):
        var next_position = self.position + self.direction
        # Death detection
        var active_death = next_position in self or not Snake.in_bounds(self.position)
        var passive_death = self.fitness <= (self.score - INITIAL_SCORE) * TTL
        if active_death or passive_death:
            # Active death: If the snake hits the game bounds or tail
            self.direction = Vector3D.splat(0)
            return

        # Food detection
        if next_position == fruit_position:
            self.score += 1
            self.fitness = (self.score - INITIAL_SCORE + 1) * TTL# + ((self.score - INITIAL_SCORE) * (TTL // 8))
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
    fn draw(borrowed self, screen: PythonObject, old_fitness: Vector1D, best_fitness: Vector1D) raises:
        var pygame = Python.import_module("pygame")
        var font = pygame.font.SysFont('Comic Sans MS', 20)
        # Draw the body
        var count = 0
        var color = (200, 30, 30)
        for b in self.history:
            var body_part = b[]
            var x_position = body_part[0].to_int() + GAME_WIDTH_OFFSET
            var y_position = body_part[1].to_int() + GAME_WIDTH_OFFSET
            var z_position = body_part[2].to_int() + GAME_DEPTH_OFFSET
            var z_value = clamp(z_position / GAME_DEPTH, 0, 1)
            var rect = (x_position * GAME_SCALE, y_position * GAME_SCALE, GAME_SCALE, GAME_SCALE)
            var weight = clamp(self.fitness / best_fitness * 200, 0, 200).to_int()
            var weight_z = clamp(z_value * 200, 0, 200).to_int()
            var body_part_color = (200 - weight_z, weight, 20)
            pygame.draw.rect(screen, body_part_color, rect)
            if count == self.score - 1:
                var text_surface = font.render(str(self.fitness), False, (255, 255, 255))
                screen.blit(text_surface, (x_position * GAME_SCALE - GAME_SCALE, y_position * GAME_SCALE))
                
            count += 1
