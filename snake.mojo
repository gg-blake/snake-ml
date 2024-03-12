from python import Python
from algorithm.sort import sort
from population import dtype, nn_dtype, game_width, game_width_offset, game_height, game_height_offset, starting_score, game_scale, Population
from tensor import Tensor, TensorShape
from collections import Set
from key_position import KeyPosition
from logger import Logger

@value
struct Snake(CollectionElement, Hashable):
    var id: Int
    var pos_ptr: Pointer[Int]
    var score: Int
    var active: Bool
    var min_dist: Float32
    var neural_network: nn_dtype
    var body: PythonObject
    var body_set: Set[KeyPosition]

    fn __init__(inout self, id: Int):
        try:
            var collections = Python.import_module("collections")
            self.body = collections.deque()
        except:
            var collections = Python.none()
            self.body = Python.none()
        self.id = id
        self.pos_ptr = Pointer[Int].alloc(2)
        self.pos_ptr.store(0, 1)
        self.pos_ptr.store(1, 0)
        self.score = starting_score
        self.active = True
        self.min_dist = game_width * game_height
        self.neural_network = nn_dtype(id)
        
        self.body_set = Set[KeyPosition]()

    fn __init__(inout self, nn_data: nn_dtype, id: Int):
        self.__init__(id)
        self.neural_network = nn_data

    fn __copyinit__(inout self, existing: Self):
        self = Self(existing.neural_network, existing.id)
        self.pos_ptr = existing.pos_ptr
        self.body = existing.body
        


    fn __moveinit__(inout self, owned existing: Self):
        self = existing


    fn __hash__(self) -> Int:
        return self.id

    fn clone(inout self: Self, existing: Self):
        self = Snake(existing.neural_network, existing.id)

    fn __eq__(self, other: Snake) -> Bool:
        return hash(self) == hash(other)

    fn inBounds(self) -> Int:
        if (self.pos_ptr.load[Int](0) >= -game_width_offset and self.pos_ptr.load[Int](0) <= game_width_offset and self.pos_ptr.load[Int](1) >= -game_height_offset and self.pos_ptr.load[Int](1) <= game_height_offset):
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
            if key.x == self.pos_ptr.load[Int](0) and key.y == self.pos_ptr.load[Int](1):
                count += 1

        if count > 1:
            return 1
        
        return 0

    @staticmethod
    fn inBody(body_set: Set[KeyPosition], x: Int, y: Int) -> Int:
        var count = 0
        for key_ref in body_set:
            var key = key_ref[]
            if key.x == x and key.y == y:
                count += 1

        if count > 1:
            return 1
        
        return 0

    fn move(inout self, x: Int, y: Int) raises:
        

        var key = KeyPosition(self.pos_ptr.load[Int](0), self.pos_ptr.load[Int](1))
        var tuple = self.pos_ptr.load[Int](0), self.pos_ptr.load[Int](1)
        
        _ = self.body.appendleft(tuple)
        self.body_set.add(key)

        if len(self.body) > self.score:
            print(self.body)
            var tail_tuple = self.body.pop()
            var tail_key = KeyPosition(tail_tuple[0].to_float64().to_int(), tail_tuple[1].to_float64().to_int())
            print("removing", tail_key.__str__())
            self.body_set.remove(tail_key)

        self.pos_ptr.store[Int](0, x)
        self.pos_ptr.store[Int](1, y)

        

    fn update(inout self, x_position_fruit: Int, y_position_fruit: Int):
        if self.inBody() or not self.inBounds():
            self.active = False
        elif self.pos_ptr.load[Int](0) == x_position_fruit and self.pos_ptr.load[Int](1) == y_position_fruit:
            self.score += 1
            self.min_dist = game_width * game_height

    fn think(inout self, x_position_fruit: Int, y_position_fruit: Int) raises:
        var torch = Python.import_module("torch")
        if torch.cuda.is_available():
            var device = torch.device("cuda")
        else:
            var device = torch.device("cpu")
        if not self.active:
            return

        var input_tensor = torch.tensor([self.pos_ptr.load[Int](0), self.pos_ptr.load[Int](1), x_position_fruit, y_position_fruit, 
                                        Snake.inBounds(self.pos_ptr.load[Int](0) - 1, self.pos_ptr.load[Int](1)), 
                                        Snake.inBounds(self.pos_ptr.load[Int](0) + 1, self.pos_ptr.load[Int](1)), 
                                        Snake.inBounds(self.pos_ptr.load[Int](0), self.pos_ptr.load[Int](1) - 1), 
                                        Snake.inBounds(self.pos_ptr.load[Int](0), self.pos_ptr.load[Int](1) + 1),
                                        Snake.inBody(self.body_set, self.pos_ptr.load[Int](0) - 1, self.pos_ptr.load[Int](1)), 
                                        Snake.inBody(self.body_set, self.pos_ptr.load[Int](0) + 1, self.pos_ptr.load[Int](1)), 
                                        Snake.inBody(self.body_set, self.pos_ptr.load[Int](0), self.pos_ptr.load[Int](1) - 1), 
                                        Snake.inBody(self.body_set, self.pos_ptr.load[Int](0), self.pos_ptr.load[Int](1) + 1)]).to(torch.float32).unsqueeze(1)

        var output_tensor = self.neural_network.feed(input_tensor)
        
        _ = torch.flatten(output_tensor)
        var potential_body_parts = VariadicList((self.pos_ptr.load[Int](0), self.pos_ptr.load[Int](1) + 1), (self.pos_ptr.load[Int](0) - 1, self.pos_ptr.load[Int](1)), (self.pos_ptr.load[Int](0), self.pos_ptr.load[Int](1) - 1), (self.pos_ptr.load[Int](0) + 1, self.pos_ptr.load[Int](1)))
        for index in range(len(potential_body_parts)):
            var position = KeyPosition(potential_body_parts[index].get[0, Int](), potential_body_parts[index].get[1, Int]())
            if position in self.body_set:
                _ = output_tensor.__setitem__(index, 0, -2)
        
        var choice = torch.argmax(output_tensor).item()
        


        if choice == 0:
            self.move(self.pos_ptr[0], self.pos_ptr[1] - 1)
        elif choice == 1:
            self.move(self.pos_ptr[0] + 1, self.pos_ptr[1])
        elif choice == 2:
            self.move(self.pos_ptr[0], self.pos_ptr[1] + 1)
        elif choice == 3:
            self.move(self.pos_ptr[0] - 1, self.pos_ptr[1])

        
            

        self.update(x_position_fruit, y_position_fruit)

    # Draws visual representation of this Snake object to the running pygame window
    def draw(self, current_food_count: Int, inout screen: PythonObject):
        var pygame = Python.import_module("pygame")
        # Snakes that have eaten the most food off their generation are the brightest
        var true_score = self.score - starting_score
        main_weight = int((true_score + 1) / current_food_count * 255)
        
        # Draw the body
        count = 0
        for key_ref in self.body_set:
            var key = key_ref[]
            count += 1
            try:
                # Points in the body get darker the closer they are to the end
                tail_weight = int(count / len(self.body) * 32 + int((true_score + 1) / current_food_count * 128))
                # Draw rect to screen
                if self.score - starting_score + 1 >= current_food_count:
                    pygame.draw.rect(screen, (200, 30, 30), ((key.x + game_width_offset) * game_scale, (key.y + game_height_offset) * game_scale, game_scale, game_scale))
                else:
                    pygame.draw.rect(screen, (200, 0, 0), (key.x + game_width_offset) * game_scale, (key.y + game_height_offset) * game_scale, game_scale, game_scale)
            except ValueError:
                pass
            
        # Draw the head of the snake
        try:
            if true_score + 1 >= current_food_count:
                pygame.draw.rect(screen, (120, 255, 120), ((self.pos_ptr.load[Int](0) + game_width_offset) * game_scale, (self.pos_ptr.load[Int](1) + game_height_offset) * game_scale, game_scale, game_scale))
            else:
                pygame.draw.rect(screen, (60, 60, 60), ((self.pos_ptr.load[Int](0) + game_width_offset) * game_scale, (self.pos_ptr.load[Int](1) + game_height_offset) * game_scale, game_scale, game_scale))
        except ValueError:
            pass
    