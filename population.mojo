from python import Python
from snake import Snake
from collections import Set, Dict
from neural_network import NeuralNetwork
from algorithm.sort import sort
from logger import Logger
from key_position import KeyTuple

alias dtype = DType.float32
alias nn_dtype = NeuralNetwork[12, 16, 4]
alias game_width: Int = 40
alias game_width_offset: Int = game_width // 2
alias game_height: Int = 40
alias game_height_offset: Int = game_height // 2
alias starting_score: Int = 5
alias game_scale: Int = 13

struct Population[snake_count: Int]:
    var habitat: Dict[Int, Snake]
    var food_array: DynamicVector[KeyTuple]
    var active: Bool
    var generation: Int
    var clock: PythonObject

    fn __init__(inout self) raises:
        var pygame = Python.import_module("pygame")
        _ = pygame.init()
        _ = pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.habitat = Dict[Int, Snake]()
        self.food_array = DynamicVector[KeyTuple]()
        self.active = True
        self.generation = 0
        self.generate_food()
        for id in range(snake_count):
            self.habitat[id] = Snake(id)

        

    fn generate_food(inout self) raises:
        var pyrandom = Python.import_module("random")
        var collections = Python.import_module("collections")
        var Point = collections.namedtuple('Point', ['x', 'y'])
        var rand_x = pyrandom.randint(-game_width_offset, game_width_offset).to_float64().to_int()
        var rand_y = pyrandom.randint(-game_height_offset, game_height_offset).to_float64().to_int()
        self.food_array.append(KeyTuple(rand_x, rand_y, Point))

    fn update_habitat(inout self, inout screen: PythonObject) raises -> Bool:
        var pygame = Python.import_module("pygame")
        var survived = False
        

        _ = screen.fill((0, 0, 0))
        var index = 0
        for index in range(len(self.habitat)):
            
            if self.habitat[index].score - starting_score >= len(self.food_array):
                self.generate_food()
            var current_snake_fruit = self.food_array[self.habitat[index].score - starting_score]
            print("Start", self.habitat[index].id)
            self.habitat[index].think(current_snake_fruit.x, current_snake_fruit.y)
            print("End", self.habitat[index].id)

            var distance: Float32 = Population[snake_count].euclidean_distance(self.habitat[index].x, 
                self.habitat[index].y, 
                current_snake_fruit.x, 
                current_snake_fruit.y)

            if distance < self.habitat[index].min_dist:
                self.habitat[index].min_dist = distance

            if self.habitat[index].active:
                survived = True

            _ = self.habitat[index].draw(len(self.food_array), screen)

        
        print("end of updating.")
        self.draw_food(screen)
        
        _ = pygame.display.update()
        _ = self.clock.tick(60)

        return survived

    @staticmethod
    fn fitness(snake: Snake) -> Float32:
        return ((400 / (snake.min_dist + 1)) + (snake.score**2 * 20))

    @staticmethod
    fn euclidean_distance(x1: Int, y1: Int, x2: Int, y2: Int) -> Float32:
        return ((x1 - x2)**2 + (y1 - y2)**2)**(0.5)

    fn generate_next_habitat(inout self, survival_rate: Float32) raises:
        # I don't know how to implement Mojo built-in sort so this is my work around :(
        var fitness_dict = Dict[Int, Int]()
        var fitness_array = DynamicVector[Int]()
        var index = 0
        for snake_reference in self.habitat:
            var current_snake: Snake = snake_reference[]
            var current_snake_fitness: Int = Population[snake_count].fitness(current_snake).to_int()

            fitness_dict[current_snake_fitness] = index
            fitness_array.append(current_snake_fitness)
            index += 1

        sort(fitness_array)

        # Resurrect the fittest snakes, their objects are preserved in the habitat
        for index in range(snake_count):
            var habitat_index = fitness_dict[fitness_array[index]]
            self.habitat[habitat_index].active = True

        # For every two fit snakes, create two children with blended weights and biases and replace two unfit snake objects in the habitat
        for index in range(snake_count * survival_rate, snake_count, 2):
            var habitat_index = fitness_dict[fitness_array[index]]
            var fit_traits_a: nn_dtype = self.habitat[fitness_dict[fitness_array[index - (snake_count // 2)]]].neural_network
            var fit_traits_b: nn_dtype = self.habitat[fitness_dict[fitness_array[index - (snake_count // 2) + 1]]].neural_network
            var new_traits: nn_dtype = NeuralNetwork.blend_genetic_traits(fit_traits_a, fit_traits_b, self.generation * snake_count + index)
            var child: Snake = Snake(new_traits, self.generation * snake_count + index)
            self.habitat[habitat_index] = child
            self.habitat[habitat_index + 1] = child
            self.habitat[habitat_index + 1].id += 1

    fn draw_food(self, inout screen: PythonObject) raises:
        
        var pygame = Python.import_module("pygame")
        print("Drawing")
        var last_food_x = self.food_array[-1].x + game_width_offset
        var last_food_y = self.food_array[-1].y + game_height_offset
        _ = pygame.draw.rect(screen, (0, 100, 0), (last_food_x * game_scale, last_food_y * game_scale, game_scale, game_scale))
        print("Drawn")
        if len(self.food_array) <= 1:
            return

        for index in range(0, len(self.food_array) - 1):
            print("Drawing food...", index)
            var food = self.food_array[index]
            var food_x = food.x + game_width_offset
            var food_y = food.y + game_height_offset
            # Draws visual representation of this Food object to the running pygame window
            _ = pygame.draw.rect(screen, (0, 200, 0), (food_x * game_scale, food_y * game_scale, game_scale, game_scale))
            

        
    