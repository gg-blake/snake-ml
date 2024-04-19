from python import Python
from population import dtype, neural_network_spec, game_width, game_width_offset, game_height, game_height_offset, starting_score, game_scale
from neural_network import NeuralNetwork
from math import abs, sqrt
from tensor import Tensor, TensorSpec

struct Snake(Hashable):
    var position: SIMD[dtype, 2]
    var direction: SIMD[dtype, 2]
    var score: Int
    var min_dist: SIMD[dtype, 1]
    var neural_network: NeuralNetwork[dtype]
    var history: List[SIMD[dtype, 2]]

    fn __init__(inout self) raises:
        self.position = SIMD[dtype, 2](0, 0)
        self.direction = SIMD[dtype, 2](-1, 0)
        self.neural_network = NeuralNetwork[dtype](spec=neural_network_spec)
        self.score = starting_score
        self.min_dist = game_width * game_height
        self.history = List[SIMD[dtype, 2]]()
        for i in range(self.score):
            self.history.append(self.position + SIMD[dtype, 2](self.score - i - 1, 0))

    # Make a Snake instance and transfer ownership of NeuralNetwork
    fn __init__(inout self, owned neural_network: NeuralNetwork[dtype]):
        self.position = SIMD[dtype, 2](0, 0)
        self.direction = SIMD[dtype, 2](-1, 0)
        self.neural_network = neural_network
        self.score = starting_score
        self.min_dist = game_width * game_height
        self.history = List[SIMD[dtype, 2]]()
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

    fn update(inout self, fruit_position: SIMD[dtype, 2], borrowed food_array_length: Int, inout screen: PythonObject) raises:
        var torch = Python.import_module("torch")
        var dist_left = abs(-game_width_offset - self.position[0])
        var dist_right = abs(game_width_offset - self.position[0])
        var dist_top = abs(-game_height_offset - self.position[1])
        var dist_bottom = abs(game_height_offset - self.position[1])
        var dist_top_left = sqrt(dist_top**2 + dist_left**2)
        var dist_top_right = sqrt(dist_top**2 + dist_right**2)
        var dist_bottom_left = sqrt(dist_bottom**2 + dist_left**2)
        var dist_bottom_right = sqrt(dist_bottom**2 + dist_right**2)
        var fruit_left = (fruit_position < self.position)[0]
        var fruit_right = (fruit_position > self.position)[0]
        var fruit_top = (fruit_position < self.position)[1]
        var fruit_bottom = (fruit_position > self.position)[1]

        var fruit_left_strict = SIMD[DType.bool, 1](fruit_left and not (fruit_top or fruit_bottom)).to_int()
        var fruit_right_strict = SIMD[DType.bool, 1](fruit_right and not (fruit_top or fruit_bottom)).to_int()
        var fruit_top_strict = SIMD[DType.bool, 1](fruit_top and not (fruit_left or fruit_right)).to_int()
        var fruit_bottom_strict = SIMD[DType.bool, 1](fruit_bottom and not (fruit_left or fruit_right)).to_int()
        var body_left = (self.position + SIMD[dtype, 2](-1, 0) in self).to_int()
        var body_right = (self.position + SIMD[dtype, 2](1, 0) in self).to_int()
        var body_top = (self.position + SIMD[dtype, 2](0, -1) in self).to_int()
        var body_bottom = (self.position + SIMD[dtype, 2](0, 1) in self).to_int()

        var input = torch.tensor([
            dist_left.to_int(), dist_right.to_int(), dist_top.to_int(), dist_bottom.to_int(),
            dist_top_left.to_int(), dist_top_right.to_int(), dist_bottom_left.to_int(), dist_bottom_right.to_int(),
            fruit_left_strict, fruit_right_strict, fruit_top_strict, fruit_bottom_strict,
            (fruit_left and fruit_top).to_int(), (fruit_right and fruit_top).to_int(),
            (fruit_left and fruit_bottom).to_int(), (fruit_right and fruit_bottom).to_int(),
            body_left, body_right, body_top, body_bottom
        ]).to(torch.float32).unsqueeze(1)

        

        var output = self.neural_network.feed(input)
        output = torch.flatten(output)
        var sorted: PythonObject
        var indices: PythonObject
        output = torch.sort(output)[1]
        output = output.flip(0)

        for control in output:
            if control == 0:
                if self.direction == SIMD[dtype, 2](0, 1):
                    continue
                # Left
                self.direction = SIMD[dtype, 2](0, -1)
                break
            elif control == 1:
                if self.direction == SIMD[dtype, 2](0, -1):
                    continue
                # Right
                self.direction = SIMD[dtype, 2](0, 1)
                break
            elif control == 2:
                if self.direction == SIMD[dtype, 2](-1, 0):
                    continue
                # Up
                self.direction = SIMD[dtype, 2](1, 0)
                break
            elif control == 3:
                if self.direction == SIMD[dtype, 2](1, 0):
                    continue
                # Down
                self.direction = SIMD[dtype, 2](-1, 0)
                break

        #print(self.position, self.direction, self.position + self.direction)

        

        
        self.move(fruit_position)
        self.draw(food_array_length, screen)
        #print(self.position)

    fn in_bounds(self) -> Bool:
        return abs(self.position[0]) <= game_width_offset and abs(self.position[1]) <= game_height_offset

    fn distance(self, point: SIMD[dtype, 2]) -> SIMD[dtype, 1]:
        return sqrt((self.position[0] - point[0])**2 + (self.position[1] - point[1])**2)

    fn move(inout self, fruit_position: SIMD[dtype, 2]):
        if self.position + self.direction in self or not self.in_bounds():
            self.direction = SIMD[dtype, 2].splat(0)
            return
        
        if self.position + self.direction == fruit_position:
            self.score += 1
            self.min_dist = game_width * game_height
        else:
            var current_distance = self.distance(fruit_position)
            if current_distance < self.min_dist:
                self.min_dist = current_distance

        self.position += self.direction
        self.history = self.history[1:]
        self.history.append(self.position)
        


    fn fitness(self) -> SIMD[dtype, 1]:
        return ((400 / (self.min_dist) + (self.score**2 * 20)))

    fn generate_offspring(inout parent_a: Snake, inout parent_b: Snake) raises:
        parent_a.neural_network.average(parent_b.neural_network)
        parent_b.neural_network.average(parent_b.neural_network)

    # Draws visual representation of this Snake object to the running pygame window
    fn draw(borrowed self, current_food_count: Int, screen: PythonObject):
        var pygame = Python.none()
        try:
            pygame = Python.import_module("pygame")
        except Error:
            print(Error)
        # Snakes that have eaten the most food off their generation are the brightest
        var true_score = self.score - starting_score
        var main_weight = int((true_score + 1) / current_food_count * 255)
        
        # Draw the body
        var count = 0
        
        

        

        for ref in self.history:
            var key = ref.get_unsafe_pointer()[]
            
            
            try:
                
                
                # Points in the body get darker the closer they are to the end
                var tail_weight = int(count / len(self.history) * 32 + int((true_score + 1) / current_food_count * 128))
                # Draw rect to screen
                if self.score - starting_score + 1 >= current_food_count:
                    pygame.draw.rect(screen, (200, 30, 30), ((key[0].to_int() + game_width_offset) * game_scale, (key[1].to_int() + game_height_offset) * game_scale, game_scale, game_scale))
                else:
                    pygame.draw.rect(screen, (200, 0, 0), (key[0].to_int() + game_width_offset) * game_scale, (key[1].to_int() + game_height_offset) * game_scale, game_scale, game_scale)
                
            except ValueError:
                pass

            count += 1
            
            
        # Draw the head of the snake
        try:
            if true_score + 1 >= current_food_count:
                pygame.draw.rect(screen, (120, 255, 120), ((self.position[0].to_int() + game_width_offset) * game_scale, (self.position[1].to_int() + game_height_offset) * game_scale, game_scale, game_scale))
            else:
                pass
                pygame.draw.rect(screen, (60, 60, 60), ((self.position[0].to_int() + game_width_offset) * game_scale, (self.position[1].to_int() + game_height_offset) * game_scale, game_scale, game_scale))
        except ValueError:
            pass



    
    