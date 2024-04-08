from python import Python
from algorithm.sort import sort
from population import dtype, nn_dtype, game_width, game_width_offset, game_height, game_height_offset, starting_score, game_scale, Population
from tensor import Tensor, TensorShape
from collections import Set
from key_position import KeyTuple
from logger import Logger


struct Snake(CollectionElement, Hashable):
    var id: Int
    var x: Int
    var y: Int
    var score: Int
    var active: Bool
    var min_dist: Float32
    var neural_network: nn_dtype
    var body: PythonObject
    var body_set: Set[Int]

    fn __init__(inout self, id: Int):
        self.id = id
        self.x = 0
        self.y = 0
        self.score = starting_score
        self.active = True
        self.min_dist = game_width * game_height
        self.neural_network = nn_dtype(id)
        self.body_set = Set[Int]()
        try:
            var collections = Python.import_module("collections")
            self.body = collections.deque()
        except:
            var collections = Python.none()
            self.body = Python.none()
        
        

    fn __init__(inout self, nn_data: nn_dtype, id: Int):
        self.__init__(id)
        self.neural_network = nn_data

    fn __copyinit__(inout self, existing: Self):
        self = Self(existing.id)
        self.body_set = Set[Int](existing.body_set)
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
        self.body_set = Set[Int](existing.body_set)
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
            var key = key_ref[]
            if KeyTuple(self.x, self.y).hash == key:
                count += 1

        if count > 1:
            return 1
        
        return 0

    @staticmethod
    fn inBody(body_set: Set[Int], x: Int, y: Int) -> Int:
        var count = 0
        for key_ref in body_set:
            var key: Int = key_ref[]
            if KeyTuple(x, y).hash == key:
                count += 1

        if count > 1:
            return 1
        
        return 0

    # DEBUG TOOL #
    fn print_body_set(self):
        print("{")
        for ref in self.body_set:
            print(ref[], end="")
        print("}")

    fn move(inout self, x: Int, y: Int) raises:

        var key = KeyTuple(self.x, self.y)
        var tuple = self.x, self.y
        
        _ = self.body.appendleft(tuple)
        self.body_set.add(key.hash)

        if len(self.body) > self.score:
            var tail_tuple = self.body.pop()
            var tail_key = KeyTuple(tail_tuple[0].to_float64().to_int(), tail_tuple[1].to_float64().to_int())
            #print("removing", tail_key.__str__())
            try:
                self.body_set.remove(tail_key.hash)
            except:
                pass

        self.x = x
        self.y = y

        

    fn update(inout self, fruit_pos: KeyTuple):
        if self.inBody() or not self.inBounds():
            self.active = False
        elif KeyTuple(self.x, self.y) == fruit_pos:
            self.score += 1
            self.min_dist = game_width * game_height

    fn distance(self, point: KeyTuple) raises -> Float64:
        return KeyTuple(self.x, self.y).distance(point)

    fn think(inout self, fruit_pos: KeyTuple) raises:
        var torch = Python.import_module("torch")
        if torch.cuda.is_available():
            var device = torch.device("cuda")
        else:
            var device = torch.device("cpu")
        if not self.active:
            return

        

        var input_tensor = torch.tensor([self.x, self.y, fruit_pos[0], fruit_pos[1], 
                                        Snake.inBounds(self.x - 1, self.y), 
                                        Snake.inBounds(self.x + 1, self.y), 
                                        Snake.inBounds(self.x, self.y - 1), 
                                        Snake.inBounds(self.x, self.y + 1),
                                        Snake.inBody(self.body_set, self.x - 1, self.y), 
                                        Snake.inBody(self.body_set, self.x + 1, self.y), 
                                        Snake.inBody(self.body_set, self.x, self.y - 1), 
                                        Snake.inBody(self.body_set, self.x, self.y + 1)]).to(torch.float32).unsqueeze(1)

        var output_tensor = self.neural_network.feed(input_tensor)
        
        

        _ = torch.flatten(output_tensor)
        var potential_body_parts = List(List(self.x, self.y + 1), List(self.x - 1, self.y), List(self.x, self.y - 1), List(self.x + 1, self.y))
        for index in range(len(potential_body_parts)):
            var position = KeyTuple(potential_body_parts[index][0], potential_body_parts[index][1])
            if position.hash in self.body_set:
                _ = output_tensor.__setitem__(index, -2)
        
        var choice = torch.argmax(output_tensor).item()
        # {(0, 1), (-1, 1), (-1, 2), (-2, 2), (-2, 3), (-3, 3)}
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
    fn draw(self, current_food_count: Int, inout screen: PythonObject):
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
        
        ### TODO: Iterate through the python deque if possible but need to iterate through something other than self.body_set (its no longer useable)
        for key_ref in self.body_set:
            var key = key_ref[]
            count += 1
            try:
                var tmp_key = KeyTuple(self.x, self.y)
                # Points in the body get darker the closer they are to the end
                var tail_weight = int(count / len(self.body) * 32 + int((true_score + 1) / current_food_count * 128))
                # Draw rect to screen
                if self.score - starting_score + 1 >= current_food_count:
                    pygame.draw.rect(screen, (200, 30, 30), ((tmp_key[0] + game_width_offset) * game_scale, (tmp_key[1] + game_height_offset) * game_scale, game_scale, game_scale))
                else:
                    pygame.draw.rect(screen, (200, 0, 0), (tmp_key[0] + game_width_offset) * game_scale, (tmp_key[1] + game_height_offset) * game_scale, game_scale, game_scale)
            except ValueError:
                pass
            
        # Draw the head of the snake
        try:
            if true_score + 1 >= current_food_count:
                pygame.draw.rect(screen, (120, 255, 120), ((self.x + game_width_offset) * game_scale, (self.y + game_height_offset) * game_scale, game_scale, game_scale))
            else:
                pygame.draw.rect(screen, (60, 60, 60), ((self.x + game_width_offset) * game_scale, (self.y + game_height_offset) * game_scale, game_scale, game_scale))
        except ValueError:
            pass


    