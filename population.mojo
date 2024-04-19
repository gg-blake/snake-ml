from python import Python
from snake import Snake
from collections import Set, Dict
from collections.optional import Optional
from neural_network import NeuralNetwork
from algorithm.sort import sort, partition
from algorithm.functional import parallelize
from math import sqrt, abs, floor
from tensor import Tensor
from collections.vector import InlinedFixedVector
from logger import Logger

alias dtype = DType.float32
alias neural_network_spec = List[Int](20, 35, 27, 10, 4)
alias game_width: Int = 40
alias game_width_offset: Int = game_width // 2
alias game_height: Int = 40
alias game_height_offset: Int = game_height // 2
alias starting_score: Int = 5
alias game_scale: Int = 13

struct Population[snake_count: Int]:
    var habitat: AnyPointer[Snake]
    var food_array: List[SIMD[dtype, 2]]
    var active: Bool
    var generation: Int
    var screen: PythonObject

    fn __init__(inout self) raises:
        var pygame = Python.import_module("pygame")
        pygame.init()
        pygame.display.set_caption("Snake AI")
        self.screen = pygame.display.set_mode((game_width * game_scale, game_height * game_scale))
        self.habitat = AnyPointer[Snake].alloc(snake_count)
        for id in range(snake_count):
            self.habitat[id] = Snake()
        
        self.food_array = List[SIMD[dtype, 2]]()
        self.active = True
        self.generation = 0
        self.generate_food()

    fn __getitem__(inout self, idx: Int) raises -> NeuralNetwork[dtype]:
        if idx >= snake_count or idx < 0:
            raise Error("Habitat index out of range")

        return self.habitat[idx].neural_network

    fn __getitem__(inout self, idx: Int, idx_layer: Int) raises -> PythonObject:
        return self.habitat[idx].neural_network[idx_layer]

    fn __getitem__(inout self, idx: Int, idx_layer: Int, idx_row: Int) raises -> PythonObject:
        return self.habitat[idx].neural_network[idx_layer, idx_row]

    fn __getitem__(inout self, idx: Int, idx_layer: Int, idx_row: Int, idx_col: Int) raises -> SIMD[dtype, 1]:
        return self.habitat[idx].neural_network[idx_layer, idx_row, idx_col]

    fn generate_food(inout self) raises:
        var pyrandom = Python.import_module("random")
        var collections = Python.import_module("collections")
        var rand_x = pyrandom.randint(-game_width_offset, game_width_offset).to_float64().to_int()
        var rand_y = pyrandom.randint(-game_height_offset, game_height_offset).to_float64().to_int()
        self.food_array.append(SIMD[dtype, 2](rand_x, rand_y))

    fn update_habitat(inout self) raises:
        Logger.cls()
        Logger.status("Updating habitat.")
        var pygame = Python.import_module("pygame")

        self.screen.fill((0, 0, 0))

        self.active = False
        for index in range(snake_count):
            if self.habitat[index].score - starting_score >= len(self.food_array):
                self.generate_food()
            
            var current_snake_fruit = self.food_array[self.habitat[index].score - starting_score]
            self.habitat[index].update(current_snake_fruit, len(self.food_array), self.screen)
        
            if self.habitat[index].direction[0] != 0 or self.habitat[index].direction[1] != 0:
                self.active = True
            
        self.draw_food(self.screen)
        pygame.display.update()

    fn generate_next_habitat(inout self, survival_rate: Float32) raises:
        Logger.notice("Generating new habitat.")
        self.active = True
        #var new_habitat: List[Snake] = List[Snake]()
        var snake_fitnesses: List[Float32] = List[Float32]()


        for index in range(0, snake_count):
            snake_fitnesses.append(self.habitat[index].fitness())

        sort(snake_fitnesses) # Sort in ascending order (low to high)



        var k = floor(snake_count - (snake_count * survival_rate)).to_int() # Reverse list index
        var survival_threshold = snake_fitnesses[k] # Top k snakes live to next habitat
        var parent_threshold = snake_fitnesses[-2] # Top two become parents
        var parent_a: Optional[Int] = Optional[Int](None)
        var parent_b: Optional[Int] = Optional[Int](None)
        #var child: Optional[Snake] = Optional[Snake](None)

        if len(self.food_array) > 1:
            self.food_array = List(self.food_array[-1])

        '''var indices = Pointer[Int].alloc(snake_count)
        for i in range(snake_count):
            indices[i] = i

        @parameter
        fn cmp_snake[type: AnyRegType](a: Int, b: Int) -> Bool:
            return True

        partition[Int, cmp_snake](snake_count * survival_rate, snake_count)'''



        for index in range(0, snake_count, 2):
            self.habitat[index] = Snake(self.habitat[index].neural_network)
            self.habitat[index+1] = Snake(self.habitat[index+1].neural_network)
            self.habitat[index].neural_network.average(self.habitat[index+1].neural_network)
            self.habitat[index+1].neural_network.average(self.habitat[index].neural_network)
            self.habitat[index].neural_network.mutate(0.5)
            self.habitat[index+1].neural_network.mutate(0.5)
        '''for index in range(0, snake_count):
            self.habitat[index] = Snake(nn_data=self.habitat[index].neural_network)
            
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

        self.habitat = new_habitat'''
            
    fn draw_food(self, screen: PythonObject) raises:
        var pygame = Python.import_module("pygame")
        var last_food_x = self.food_array[-1][0] + game_width_offset
        var last_food_y = self.food_array[-1][1] + game_height_offset
        pygame.draw.rect(screen, (0, 100, 0), (last_food_x.to_int() * game_scale, last_food_y.to_int() * game_scale, game_scale, game_scale))
        if len(self.food_array) <= 1:
            return

        for index in range(0, len(self.food_array) - 1):
            var food = self.food_array[index]
            var food_x = food[0] + game_width_offset
            var food_y = food[1] + game_height_offset
            # Draws visual representation of this Food object to the running pygame window
            pygame.draw.rect(screen, (0, 200, 0), (int(food_x) * game_scale, int(food_y) * game_scale, game_scale, game_scale))

    fn save(inout self) raises:
        for habitat_index in range(snake_count):
            self.habitat[habitat_index].neural_network.save(habitat_index)

        var filename_prefix = "data/" + str(self.habitat[0].neural_network.__repr__())
        Logger.notice("Population data serialized as " + filename_prefix + "-#-#")

    fn load(inout self) raises:
        for habitat_index in range(snake_count):
            self.habitat[habitat_index].neural_network.load(habitat_index)

        var filename_prefix = "data/" + str(self.habitat[0].neural_network.__repr__())
        Logger.notice("Population data deserialized from " + filename_prefix + "-#-#")
        
