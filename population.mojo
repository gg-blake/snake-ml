from python import Python
from snake import Snake
from collections import Set, Dict
from neural_network import NeuralNetwork
from algorithm.sort import sort
from logger import Logger
from key_position import Position
import sys
from math import sqrt

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
    var food_array: List[Position]
    var active: Bool
    var generation: Int
    var clock: PythonObject

    fn __init__(inout self) raises:
        var pygame = Python.import_module("pygame")
        _ = pygame.init()
        _ = pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.habitat = Dict[Int, Snake]()
        self.food_array = List[Position]()
        self.active = True
        self.generation = 0
        self.generate_food()
        for id in range(snake_count):
            self.habitat[id] = Snake(id)

    fn print_locations(self):
        for ref in self.habitat.items():
            var snake: Snake = ref[].value
            print("[", snake.x, snake.y, "]", end="")

        print(",")

        

    fn generate_food(inout self) raises:
        var pyrandom = Python.import_module("random")
        var collections = Python.import_module("collections")
        var Point = collections.namedtuple('Point', ['x', 'y'])
        var rand_x = pyrandom.randint(-game_width_offset, game_width_offset).to_float64()
        var rand_y = pyrandom.randint(-game_height_offset, game_height_offset).to_float64()
        self.food_array.append(Position(rand_x, rand_y))

    fn update_habitat(inout self, inout screen: PythonObject) raises -> Bool:
        var pygame = Python.import_module("pygame")
        var survived = False
        

        _ = screen.fill((0, 0, 0))
        var index = 0
        for index in range(len(self.habitat)):
            
            if self.habitat[index].score - starting_score >= len(self.food_array):
                self.generate_food()
            var current_snake_fruit = self.food_array[self.habitat[index].score - starting_score]
            Snake.think(self.habitat[index], current_snake_fruit)

            var distance: Float32 = self.habitat[index].distance(current_snake_fruit)
            
            if distance < self.habitat[index].min_dist:
                self.habitat[index].min_dist = distance

            if self.habitat[index].active:
                survived = True

            _ = self.habitat[index].draw(len(self.food_array), screen)
            #self.print_locations()

        self.draw_food(screen)
        
        _ = pygame.display.update()
        _ = self.clock.tick(60)

        return survived

    @staticmethod
    fn fitness(snake: Snake) -> Float32:
        return ((400 / (snake.min_dist + 1)) + (snake.score**2 * 20))

    @staticmethod
    fn euclidean_distance(x1: Int, y1: Int, x2: Int, y2: Int) -> Float32:
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)

    fn generate_next_habitat(inout self, survival_rate: Float32) raises:
        # I don't know how to implement Mojo built-in sort so this is my work around :(
        self.food_array = List[Position]()
        self.generate_food()
        var fitness_dict = Dict[Int, Int]()
        var fitness_array = List[Int]()
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
        var last_food_x = self.food_array[-1][0] + game_width_offset
        var last_food_y = self.food_array[-1][1] + game_height_offset
        _ = pygame.draw.rect(screen, (0, 100, 0), (last_food_x * game_scale, last_food_y * game_scale, game_scale, game_scale))
        if len(self.food_array) <= 1:
            return

        for index in range(0, len(self.food_array) - 1):
            var food = self.food_array[index]
            var food_x = food[0] + game_width_offset
            var food_y = food[1] + game_height_offset
            # Draws visual representation of this Food object to the running pygame window
            _ = pygame.draw.rect(screen, (0, 200, 0), (food_x * game_scale, food_y * game_scale, game_scale, game_scale))
            

        
    