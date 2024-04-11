from python import Python
from algorithm.sort import sort
from population import nn_dtype, game_width, game_width_offset, game_height, game_height_offset, starting_score, game_scale, Population
from tensor import Tensor, TensorShape
from collections import Set
from key_position import Position, dtype
from logger import Logger
from sys import PrefetchRW, PrefetchLocality, PrefetchCache, PrefetchOptions
from neural_network import NeuralNetwork

struct Snake(CollectionElement, Hashable):
    var id: Int
    var x: SIMD[dtype, 1]
    var y: SIMD[dtype, 1]
    var score: Int
    var active: Bool
    var min_dist: Float32
    var neural_network: nn_dtype
    var body: List[Position]
    var body_set: Set[Position]

    fn __init__(inout self, id: Int):
        self.id = id
        self.x = 0
        self.y = 0
        self.score = starting_score
        self.active = True
        self.min_dist = game_width * game_height
        self.neural_network = nn_dtype(id)
        self.body_set = Set[Position](Position(self.x + 4, self.y), Position(self.x + 3, self.y), Position(self.x + 2, self.y), Position(self.x + 1, self.y), Position(self.x + 4, self.y))
        self.body = List(Position(self.x, self.y), Position(self.x + 1, self.y), Position(self.x + 1, self.y), Position(self.x + 3, self.y), Position(self.x + 4, self.y))

    fn __init__(inout self, nn_data: nn_dtype, id: Int):
        self.__init__(id)
        self.neural_network = nn_data

    fn __copyinit__(inout self, existing: Self):
        self = Self(existing.id)
        self.body_set = Set[Position](existing.body_set)
        self.body = existing.body
        self.id = existing.id
        self.x = existing.x
        self.y = existing.y
        self.score = existing.score
        self.active = existing.active
        self.min_dist = existing.min_dist
        self.neural_network = existing.neural_network

    fn __moveinit__(inout self, owned existing: Self):
        self = Self(existing.id)
        self.body_set = Set[Position](existing.body_set)
        self.body = existing.body
        self.id = existing.id
        self.x = existing.x
        self.y = existing.y
        self.score = existing.score
        self.active = existing.active
        self.min_dist = existing.min_dist
        self.neural_network = existing.neural_network^

    fn __hash__(self) -> Int:
        return self.id

    fn clone(inout self: Self, existing: Self):
        self = Snake(existing.neural_network, existing.id)

    fn __eq__(self: Snake, other: Snake) -> Bool:
        return hash(self) == hash(other)

    fn inBounds(self) -> Int:
        if (self.x >= -game_width_offset and self.x <= game_width_offset and self.y >= -game_height_offset and self.y <= game_height_offset):
            return 1
        
        return 0

    @staticmethod
    fn inBounds(x: Int, y: Int) -> Int:
        if (x >= -game_width_offset and x <= game_width_offset and y >= -game_height_offset and y <= game_height_offset):
            return 1
        
        return 0

    fn inBody(self) -> Int:
        var count = 0
        for key_ref in self.body_set:
            var key: Position = key_ref[]
            if Position(self.x, self.y) == key:
                count += 1

        if count > 1:
            return 1
        
        return 0

    @staticmethod
    fn inBody(body_set: Set[Position], x: Int, y: Int) -> Int:
        var count = 0
        for key_ref in body_set:
            var key: Position = key_ref[]
            if Position(x, y) == key:
                count += 1

        if count > 1:
            return 1
        
        return 0

    fn fitness(self) -> SIMD[DType.float32, 1]:
        return ((400 / (self.min_dist) + (self.score**2 * 20)))

    # DEBUG TOOL #
    fn print_body_set(self):
        print("{")
        for ref in self.body_set:
            print(ref[], end="")
        print("}")

    fn print_body(self):
        print("[")
        for ref in self.body:
            print(ref[], end="")
        print("]")

    fn move(inout self, x: SIMD[dtype, 1], y: SIMD[dtype, 1]) raises:
        var new_position = Position(x, y)
        var tmp_list = List[Position](new_position)
        tmp_list.extend(self.body)
        self.body = tmp_list
        self.body_set.add(new_position)

        var pop_position = self.body.pop_back()
        try:
            self.body_set.remove(pop_position)
        except ValueError:
            self.active = False

        if len(self.body_set) < self.score - 1:
            self.active = False
            

        self.x, self.y = x, y

    fn update(inout self, fruit_pos: Position):
        if self.inBody() or not self.inBounds():
            self.active = False
        elif Position(self.x, self.y) == fruit_pos:
            self.score += 1
            self.min_dist = game_width * game_height

    fn distance(self, point: Position) raises -> Float32:
        return Position(self.x, self.y).distance(point)

    @staticmethod
    fn generate_offspring(parent_a: Snake, parent_b: Snake) raises -> Snake:
        var child_a: Snake = Snake(nn_data=parent_a.neural_network, id=parent_a.id)
        var child_b: Snake = Snake(nn_data=parent_b.neural_network, id=parent_b.id)
        child_a.neural_network = NeuralNetwork.blend_genetic_traits(child_a.neural_network, child_b.neural_network, parent_a.id)
        child_b.neural_network = NeuralNetwork.blend_genetic_traits(child_b.neural_network, child_a.neural_network, parent_a.id)
        return child_a
        

    fn think(inout self, fruit_pos: Position) raises:
        if not self.active:
            return
            
        var torch = Python.import_module("torch")
        if torch.cuda.is_available():
            var device = torch.device("cuda")
        else:
            var device = torch.device("cpu")
    
        var input_tensor = torch.tensor([self.x.to_int(), self.y.to_int(), fruit_pos.x.to_int(), fruit_pos.y.to_int(), 
                                        Snake.inBounds(self.x.to_int() - 1, self.y.to_int()), 
                                        Snake.inBounds(self.x.to_int() + 1, self.y.to_int()), 
                                        Snake.inBounds(self.x.to_int(), self.y.to_int() - 1), 
                                        Snake.inBounds(self.x.to_int(), self.y.to_int() + 1),
                                        Snake.inBody(self.body_set, self.x.to_int() - 1, self.y.to_int()), 
                                        Snake.inBody(self.body_set, self.x.to_int() + 1, self.y.to_int()), 
                                        Snake.inBody(self.body_set, self.x.to_int(), self.y.to_int() - 1), 
                                        Snake.inBody(self.body_set, self.x.to_int(), self.y.to_int() + 1)]).to(torch.float32).unsqueeze(1)

        var output_tensor = self.neural_network.feed(input_tensor)
        
        
        
        torch.flatten(output_tensor)
        var potential_body_parts = List(List(self.x, self.y + 1), List(self.x - 1, self.y), List(self.x, self.y - 1), List(self.x + 1, self.y))
        for index in range(len(potential_body_parts)):
            var position = Position(potential_body_parts[index][0], potential_body_parts[index][1])
            if position in self.body_set:
                output_tensor.__setitem__(index, -2)
        
        var choice = torch.argmax(output_tensor).item()

        if choice == 0:
            self.move(self.x, self.y - 1)
        elif choice == 1:
            self.move(self.x + 1, self.y)
        elif choice == 2:
            self.move(self.x, self.y + 1)
        elif choice == 3:
            self.move(self.x - 1, self.y)
        
        self.update(fruit_pos)
        
        

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
        
        if not len(self.body_set) >= 1:
            return

        

        for ref in self.body_set:
            var key = ref.get_unsafe_pointer()[]
            
            
            try:
                
                
                # Points in the body get darker the closer they are to the end
                var tail_weight = int(count / len(self.body) * 32 + int((true_score + 1) / current_food_count * 128))
                # Draw rect to screen
                if self.score - starting_score + 1 >= current_food_count:
                    pygame.draw.rect(screen, (200, 30, 30), ((key.x.to_int() + game_width_offset) * game_scale, (key.y.to_int() + game_height_offset) * game_scale, game_scale, game_scale))
                else:
                    pygame.draw.rect(screen, (200, 0, 0), (key.x.to_int() + game_width_offset) * game_scale, (key.y.to_int() + game_height_offset) * game_scale, game_scale, game_scale)
                
            except ValueError:
                pass

            count += 1
            
            
        # Draw the head of the snake
        try:
            if true_score + 1 >= current_food_count:
                pygame.draw.rect(screen, (120, 255, 120), ((self.x.to_int() + game_width_offset) * game_scale, (self.y.to_int() + game_height_offset) * game_scale, game_scale, game_scale))
            else:
                pass
                pygame.draw.rect(screen, (60, 60, 60), ((self.x.to_int() + game_width_offset) * game_scale, (self.y.to_int() + game_height_offset) * game_scale, game_scale, game_scale))
        except ValueError:
            pass


    