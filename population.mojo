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
from time import sleep

alias dtype = DType.float32
alias neural_network_spec = List[Int](10, 10, 10, 3)
alias game_width: Int = 40
alias game_width_offset: Int = game_width // 2
alias game_height: Int = 40
alias game_height_offset: Int = game_height // 2
alias starting_score: Int = 5
alias game_scale: Int = 26
alias ttl: Int = 100
alias snake_count: Int = 100
alias timeout: Int = 200 # Number of snakes steps before population is automatically updated

struct Population[snake_count: Int]:
    var habitat: AnyPointer[Snake]
    var food_array: List[SIMD[dtype, 2]]
    var active: Bool
    var stats: Dict[String, Float32]
    var screen: PythonObject
    var font: PythonObject
    var logger: Logger
    var best_snake: Snake

    fn __init__(inout self) raises:
        var pygame = Python.import_module("pygame")
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Snake AI")
        self.habitat = AnyPointer[Snake].alloc(snake_count)
        self.food_array = List[SIMD[dtype, 2]]()
        self.active = True
        self.screen = pygame.display.set_mode((game_width * game_scale, game_height * game_scale))
        self.font = pygame.font.SysFont('Comic Sans MS', 20)
        self.stats = Dict[String, Float32]()
        self.stats["generation"] = 0
        self.stats["max"] = ttl
        self.stats["average"] = 0
        self.stats["median"] = 0
        self.stats["best"] = 0
        self.logger = Logger("logs")
        self.best_snake = Snake()
        self.best_snake.fitness = 0
        self.init_habitat()
        

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

    fn __del__(owned self):
        self.habitat.free()
        try:
            self.wait_for_save()
        except:
            self.logger.error("Population save unsuccessful. FeelsBadMan.")

    fn init_habitat(inout self) raises:
        self.generate_food()
        for id in range(snake_count):
            self.habitat[id] = Snake()
        try:
            self.load()
        except:
            self.logger.warn("No serialized data found. Starting new population.")

    fn generate_food(inout self) raises:
        var pyrandom = Python.import_module("random")
        var collections = Python.import_module("collections")
        var rand_x = pyrandom.randint(-game_width_offset, game_width_offset-1).to_float64().to_int()
        var rand_y = pyrandom.randint(-game_height_offset, game_height_offset-1).to_float64().to_int()
        self.food_array.append(SIMD[dtype, 2](rand_x, rand_y))

    fn update_habitat(inout self) raises:
        var pygame = Python.import_module("pygame")

        self.screen.fill((0, 0, 0))

        self.active = False
        for index in range(snake_count):
            if self.habitat[index].score - starting_score >= len(self.food_array):
                self.generate_food()
            
            var current_snake_fruit = self.food_array[self.habitat[index].score - starting_score]
            self.habitat[index].update(current_snake_fruit, len(self.food_array), self.screen, self.font, self.stats)
        
            if self.habitat[index].direction[0] != 0 or self.habitat[index].direction[1] != 0:
                self.active = True
            
        self.draw_food(self.food_array, self.screen)
        '''var game_font = pygame.font.SysFont('Comic Sans MS', 10)
        var text_surface = game_font.render("Hello World!", False, (255, 255, 255))
        self.screen.blit(text_surface, (40, 40))'''
        pygame.display.update()

    fn replay(inout self, snake: Snake) raises:
        var pygame = Python.import_module("pygame")
        var test_neural_network = NeuralNetwork[dtype](neural_network_spec)
        test_neural_network.copy(snake.neural_network)
        var test_snake = Snake(neural_network=test_neural_network)
        var active_food = self.food_array[0]
        var clock = pygame.time.Clock()
        var run = True
        while not test_snake.is_dead() and run:
            var events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    run = False
            self.screen.fill((0, 0, 0))
            var net_score = test_snake.score - starting_score
            test_snake.update(self.food_array[net_score], net_score + 1, self.screen, self.font, self.stats)
            Self.draw_food(self.food_array[0:net_score + 1], self.screen)
            pygame.display.update()
            #self.logger.status("Distance: "+str(test_snake.distance(self.food_array[net_score])))
            sleep(0.05)


            

    fn reset_habitat(inout self):
        for i in range(snake_count):
            self.habitat[i].reset()

        if len(self.food_array) > 1:
            self.food_array = List(self.food_array[-1])

        self.active = True


    fn generate_next_habitat(inout self, survival_rate: Float32, mutation_rate: Float32) raises:
        var snake_fitnesses: List[Float32] = List[Float32]()


        for index in range(0, snake_count):
            snake_fitnesses.append(self.habitat[index].fitness)

        sort(snake_fitnesses) # Sort in ascending order (low to high)

        var k = floor(snake_count - (snake_count * survival_rate)).to_int() # Reverse list index
        var survival_threshold = snake_fitnesses[k] # Top k snakes live to next habitat
        var parent_threshold = snake_fitnesses[-1] # Top two become parents
        var parent_indices = List[Int]()
        var survived_indices = List[Int]()
        var child_indices = List[Int]()

        
        
        

        for index in range(snake_count):
            if self.habitat[index].fitness >= parent_threshold.to_int() and len(parent_indices) < 1:
                parent_indices.append(index)
            elif self.habitat[index].fitness >= survival_threshold.to_int():
                survived_indices.append(index)
            else:
                child_indices.append(index)
            
        var parent_traits = NeuralNetwork[dtype](neural_network_spec)

        if self.habitat[parent_indices[0]].fitness >= self.best_snake.fitness:
            self.best_snake.neural_network.copy(self.habitat[parent_indices[0]].neural_network)

        parent_traits.copy(self.best_snake.neural_network)

        for survived_index in survived_indices:
            self.habitat[survived_index[]].neural_network.copy(parent_traits)
            self.habitat[survived_index[]].neural_network.mutate(mutation_rate)        
        
        for child_index in child_indices:
            self.habitat[child_index[]].neural_network.copy(parent_traits)
            self.habitat[child_index[]].neural_network.mutate(mutation_rate)

        if self.habitat[parent_indices[0]].fitness >= self.best_snake.fitness:
            self.best_snake.fitness = self.habitat[parent_indices[0]].fitness
            self.replay(self.habitat[parent_indices[0]])

        var calculated_median = Self.median(snake_fitnesses)
        var calculated_average = Self.average(snake_fitnesses)
        
        self.reset_habitat()
        self.log_stats(
            generation=self.stats["generation"] + 1,
            max=parent_threshold, 
            average=calculated_average,
            median=calculated_median,
            best=self.best_snake.fitness
        )


    @staticmethod
    fn average(lst: List[Float32]) -> Float32:
        var total: Float32 = 0
        for i in lst:
            total += i[]

        return total / len(lst)

    @staticmethod
    fn median(lst: List[Float32]) -> Float32:
        if len(lst) % 2 == 0:
            var median_a = lst[(len(lst) // 2) - 1]
            var median_b = lst[len(lst) // 2]
            return (median_a + median_b) / 2
        else:
            return lst[(len(lst) - 1) // 2]

    
    fn log_stats(inout self, **kwargs: Float32) raises:
        var previous_stats = self.stats
        for key in kwargs.keys():
            var current_key = key[]
            if current_key in self.stats:
                self.stats[current_key] = kwargs[current_key]

        self.logger.cls()
        for key in self.stats.keys():
            var stat = key[]
            if self.stats[stat] > previous_stats[stat]:
                self.logger.status(str(stat) + ": " + str(self.stats[stat]) + " (+)")
            else:
                self.logger.status(str(stat) + ": " + str(self.stats[stat]) + " (-)")
        

    @staticmethod
    fn draw_food(food_array: List[SIMD[dtype, 2]], screen: PythonObject) raises:
        var pygame = Python.import_module("pygame")
        var last_food_x = food_array[-1][0] + game_width_offset
        var last_food_y = food_array[-1][1] + game_height_offset
        pygame.draw.rect(screen, (0, 200, 0), (last_food_x.to_int() * game_scale, last_food_y.to_int() * game_scale, game_scale, game_scale))
        if len(food_array) <= 1:
            return

        for index in range(0, len(food_array) - 1):
            var food = food_array[index]
            var food_x = food[0] + game_width_offset
            var food_y = food[1] + game_height_offset
            # Draws visual representation of this Food object to the running pygame window
            pygame.draw.rect(screen, (0, 100, 0), (int(food_x) * game_scale, int(food_y) * game_scale, game_scale, game_scale))

    fn save(inout self) raises:
        for habitat_index in range(snake_count):
            self.habitat[habitat_index].neural_network.save(habitat_index)

        var filename_prefix = "data/" + str(self.habitat[0].neural_network.__repr__())
        self.logger.notice("Population data serialized as " + filename_prefix + "-#-#")

    fn wait_for_save(inout self) raises:
        var input = Python.import_module("builtins").input
        while True:
            var save_population = input("Save population data?(Y/n)")
            if save_population.lower()[0] == "n":
                return
            elif save_population.lower()[0] == "y":
                break
        self.save()

    fn load(inout self) raises:
        for habitat_index in range(snake_count):
            self.habitat[habitat_index].neural_network.load(habitat_index)

        var filename_prefix = "data/" + str(self.habitat[0].neural_network.__repr__())
        self.logger.notice("Population data deserialized from " + filename_prefix + "-#-#")
        
fn main() raises:
    var pygame = Python.import_module("pygame")
    var population = Population[snake_count]()
    var run = True
    while run:
        while population.active and run:
            var events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    run = False

            population.update_habitat()
        population.generate_next_habitat(
            survival_rate=0.05,
            mutation_rate=0.6
        )