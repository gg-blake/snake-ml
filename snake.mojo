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
    var x_position: Int
    var y_position: Int
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
        self.x_position = 0
        self.y_position = 0
        self.score = starting_score
        self.active = True
        self.min_dist = game_width * game_height
        self.neural_network = nn_dtype(id)
        
        self.body_set = Set[KeyPosition]()

    fn __init__(inout self, nn_data: nn_dtype, id: Int):
        self.__init__(id)
        self.neural_network = nn_data

    fn __copyinit__(inout self, existing: Self):
        self = Snake(existing.neural_network, existing.id)

    fn __moveinit__(inout self, owned existing: Self):
        self = existing


    fn __hash__(self) -> Int:
        return self.id

    fn clone(inout self: Self, existing: Self):
        self = Snake(existing.neural_network, existing.id)

    fn __eq__(self, other: Snake) -> Bool:
        return hash(self) == hash(other)

    fn inBounds(self) -> Int:
        if (self.x_position >= -game_width_offset and self.x_position <= game_width_offset and self.y_position >= -game_height_offset and self.y_position <= game_height_offset):
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
            if key.x == self.x_position and key.y == self.y_position:
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
        print(self.x_position, self.y_position)
        

        var key = KeyPosition(self.x_position, self.y_position)
        var tuple = (self.x_position, self.y_position)
        
        _ = self.body.appendleft(tuple)
        self.body_set.add(key)

        if len(self.body) > self.score:
            print(self.body)
            var tail_tuple = self.body.pop()
            var tail_key = KeyPosition(tail_tuple[0].to_float64().to_int(), tail_tuple[1].to_float64().to_int())
            self.body_set.remove(tail_key)

        self.x_position = x
        self.y_position = y

        

    fn update(inout self, x_position_fruit: Int, y_position_fruit: Int):
        if self.inBody() or not self.inBounds():
            self.active = False
        elif self.x_position == x_position_fruit and self.y_position == y_position_fruit:
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

        var input_tensor = torch.tensor([self.x_position, self.y_position, x_position_fruit, y_position_fruit, 
                                        Snake.inBounds(self.x_position - 1, self.y_position), 
                                        Snake.inBounds(self.x_position + 1, self.y_position), 
                                        Snake.inBounds(self.x_position, self.y_position - 1), 
                                        Snake.inBounds(self.x_position, self.y_position + 1),
                                        Snake.inBody(self.body_set, self.x_position - 1, self.y_position), 
                                        Snake.inBody(self.body_set, self.x_position + 1, self.y_position), 
                                        Snake.inBody(self.body_set, self.x_position, self.y_position - 1), 
                                        Snake.inBody(self.body_set, self.x_position, self.y_position + 1)]).to(torch.float32).unsqueeze(1)

        var output_tensor = self.neural_network.feed(input_tensor)
        
        _ = torch.flatten(output_tensor)
        var potential_body_parts = VariadicList((self.x_position, self.y_position + 1), (self.x_position - 1, self.y_position), (self.x_position, self.y_position - 1), (self.x_position + 1, self.y_position))
        for index in range(len(potential_body_parts)):
            var position = KeyPosition(potential_body_parts[index].get[0, Int](), potential_body_parts[index].get[1, Int]())
            if position in self.body_set:
                _ = output_tensor.__setitem__(index, 0, -2)
        
        var choice = torch.argmax(output_tensor).item()
        


        self.move(self.x_position, self.y_position + 1)

        
            

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
                pygame.draw.rect(screen, (120, 255, 120), ((self.x_position + game_width_offset) * game_scale, (self.y_position + game_height_offset) * game_scale, game_scale, game_scale))
            else:
                pygame.draw.rect(screen, (60, 60, 60), ((self.x_position + game_width_offset) * game_scale, (self.y_position + game_height_offset) * game_scale, game_scale, game_scale))
        except ValueError:
            pass

fn main():
    var snake_list = DynamicVector[Snake]()
    var snake = Snake(4)
    snake.x_position = 12
    snake.y_position = -5
    snake_list.append(snake)
    snake_list[0].x_position = 2

    