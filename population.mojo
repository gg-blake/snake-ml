from python import Python
from snake import Snake
from collections import Set, Dict
from collections.optional import Optional
from neural_network import NeuralNetwork
from algorithm.sort import sort
from logger import Logger
from key_position import Position
import sys
from math import sqrt
from logger import Logger
from time import sleep
import math

alias dtype = DType.float32
alias nn_dtype = NeuralNetwork[12, 16, 4]
alias game_width: Int = 40
alias game_width_offset: Int = game_width // 2
alias game_height: Int = 40
alias game_height_offset: Int = game_height // 2
alias starting_score: Int = 5
alias game_scale: Int = 13

struct Population[snake_count: Int]:
    var habitat: List[Snake]
    var food_array: List[Position]
    var active: Bool
    var generation: Int
    var clock: PythonObject

    fn __init__(inout self) raises:
        var pygame = Python.import_module("pygame")
        _ = pygame.init()
        _ = pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.habitat = List[Snake]()
        self.food_array = List[Position]()
        self.active = True
        self.generation = 0
        
        
        for id in range(snake_count):
            self.habitat.append(Snake(id))

        self.generate_food()
            



    fn print_locations(self):
        for ref in self.habitat:
            var snake: Snake = ref[]
            print("[", snake.x, snake.y, "]", end="")

        print(",")

        

    fn generate_food(inout self) raises:
        var pyrandom = Python.import_module("random")
        var collections = Python.import_module("collections")
        var rand_x = pyrandom.randint(-game_width_offset, game_width_offset).to_float64().to_int()
        var rand_y = pyrandom.randint(-game_height_offset, game_height_offset).to_float64().to_int()
        self.food_array.append(Position(rand_x, rand_y))

    fn update_habitat(inout self, borrowed screen: PythonObject) raises -> Bool:
        var pygame = Python.import_module("pygame")
        var survived = False

        screen.fill((0, 0, 0))
        var index = 0
        for index in range(len(self.habitat)):
            if self.habitat[index].score - starting_score >= len(self.food_array):
                self.generate_food()
            
            var current_snake_fruit = self.food_array[self.habitat[index].score - starting_score]
            var current_snake_fruit_position = Position(current_snake_fruit.x, current_snake_fruit.y)
            Snake.think(self.habitat[index], current_snake_fruit_position)
            
            var distance: Float32 = self.habitat[index].distance(current_snake_fruit_position)
            
            if distance < self.habitat[index].min_dist:
                self.habitat[index].min_dist = distance

            if self.habitat[index].active:
                survived = True
            
            self.habitat[index].draw(len(self.food_array), screen)
            
        self.draw_food(screen)
        
        pygame.display.update()
        #self.clock.tick(60)

        return survived

    @staticmethod
    fn euclidean_distance(x1: Int, y1: Int, x2: Int, y2: Int) -> Float32:
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)

    fn generate_next_habitat(inout self, survival_rate: Float32) raises:
        var new_habitat: List[Snake] = List[Snake]()
        var snake_fitnesses: List[Float32] = List[Float32]()


        for index in range(0, snake_count):
            snake_fitnesses.append(self.habitat[index].fitness())

        sort(snake_fitnesses) # Sort in ascending order (low to high)

        var k = math.floor(snake_count - (snake_count * survival_rate)).to_int() # Reverse list index
        var survival_threshold = snake_fitnesses[k] # Top k snakes live to next habitat
        var parent_threshold = snake_fitnesses[-2] # Top two become parents
        var parent_a: Optional[Int] = Optional[Int](None)
        var parent_b: Optional[Int] = Optional[Int](None)
        var child: Optional[Snake] = Optional[Snake](None)

        for index in range(0, snake_count):
            #self.habitat[index] = Snake(nn_data=self.habitat[index].neural_network, id=index)
            
            if self.habitat[index].fitness() >= parent_threshold and not parent_a:
                parent_a = index
                new_habitat.append(Snake(nn_data=self.habitat[index].neural_network, id=len(new_habitat)))
            elif self.habitat[index].fitness() >= parent_threshold and not parent_b:
                parent_b = index
                new_habitat.append(Snake(nn_data=self.habitat[index].neural_network, id=len(new_habitat)))
            elif self.habitat[index].fitness() >= parent_threshold:
                new_habitat.append(Snake(nn_data=self.habitat[index].neural_network, id=len(new_habitat)))
                child = Snake.generate_offspring(self.habitat[parent_a.value()], self.habitat[parent_b.value()])
            elif self.habitat[index].fitness() >= survival_threshold:
                new_habitat.append(Snake(nn_data=self.habitat[index].neural_network, id=len(new_habitat)))
            else:
                continue
        
        for index in range(0, snake_count):
            if self.habitat[index].fitness() < survival_threshold and child:
                var current_child_value = child.value()
                var mutated_child = current_child_value.neural_network.mutate(12)
                new_habitat.append(Snake(nn_data=mutated_child, id=len(new_habitat)))

        self.habitat = new_habitat
            




    fn draw_food(self, screen: PythonObject) raises:
        var pygame = Python.import_module("pygame")
        var last_food_x = self.food_array[-1].x + game_width_offset
        var last_food_y = self.food_array[-1].y + game_height_offset
        pygame.draw.rect(screen, (0, 100, 0), (last_food_x.to_int() * game_scale, last_food_y.to_int() * game_scale, game_scale, game_scale))
        if len(self.food_array) <= 1:
            return

        for index in range(0, len(self.food_array) - 1):
            var food = self.food_array[index]
            var food_x = food.x + game_width_offset
            var food_y = food.y + game_height_offset
            # Draws visual representation of this Food object to the running pygame window
            pygame.draw.rect(screen, (0, 200, 0), (int(food_x) * game_scale, int(food_y) * game_scale, game_scale, game_scale))
            

        
