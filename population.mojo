from python import Python
from snake import Snake
from collections import Set, Dict
from collections.optional import Optional
from neural_network import NeuralNetwork, NeuralNetworkSpec, NeuralNetworkShape
from algorithm.sort import sort, partition
from algorithm.functional import parallelize
from math import sqrt, abs, floor
from tensor import Tensor
from collections.vector import InlinedFixedVector
from logger import Logger
from time import sleep
from buffer import Buffer
from snake_handler import SnakeHandler

alias dtype = DType.float32
alias neural_network_spec = List[Int](10, 20, 20, 3)
alias SPEC = NeuralNetworkSpec(dtype, NeuralNetworkShape(10, 20, 20, 3))
alias game_width: Int = 40
alias game_width_offset: Int = game_width // 2
alias game_height: Int = 40
alias game_height_offset: Int = game_height // 2
alias starting_score: Int = 5
alias game_scale: Int = 26
alias ttl: Int = 100
alias snake_count: Int = 100
alias timeout: Int = 200 # Number of snakes steps before population is automatically updated
alias Vector2D = SIMD[dtype, 2]
alias VectorComponent = SIMD[dtype, 1]
alias PopulationStats = Dict[String, Scalar[dtype]]
alias RGB = Tuple[Int, Int, Int]

struct Population[snake_count: Int, mutation_rate: Float32]:
    var habitat: AnyPointer[Snake]
    var food_array: List[Vector2D]
    var active: Bool
    var stats: PopulationStats
    var screen: PythonObject
    var logger: Logger
    var best_snake: SnakeHandler[snake_count, mutation_rate]
    var replay_active: Bool

    fn __init__(inout self) raises:
        var pygame = Python.import_module("pygame")
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Snake AI")
        self.habitat = AnyPointer[Snake].alloc(snake_count)
        self.food_array = List[Vector2D]()
        self.active = True
        self.screen = pygame.display.set_mode((game_width * game_scale, game_height * game_scale))
        self.stats = PopulationStats()
        self.stats["generation"] = 0
        self.stats["max"] = ttl
        self.stats["average"] = 0
        self.stats["best"] = 0
        self.logger = Logger("logs")
        self.replay_active = False
        self.best_snake = SnakeHandler[snake_count, mutation_rate](Snake(), self.food_array, self.stats, self.screen)
        self.init_habitat(mutation_rate)
        

    fn __getitem__(inout self, idx: Int) raises -> NeuralNetwork[SPEC]:
        if idx >= snake_count or idx < 0:
            raise Error("Habitat index out of range")

        return self.habitat[idx].neural_network

    fn __del__(owned self):
        self.habitat.free()
        try:
            self.wait_for_save()
        except:
            self.logger.error("Population save unsuccessful. FeelsBadMan.")

    fn init_habitat(inout self, mutation_rate: Float32) raises:
        self.generate_food()
        for id in range(snake_count):
            self.habitat[id] = Snake()
        try:
            self.best_snake.load()
            self.log_stats(self.stats)
            self.best_snake.replay(0.1)
            self.stats["best"] = self.best_snake.snake.fitness
            self.load(mutation_rate)
            
            
        except:
            self.logger.warn("No serialized data found. Starting new population.")

    fn generate_food(inout self) raises:
        var pyrandom = Python.import_module("random")
        var rand_x = pyrandom.randint(-game_width_offset, game_width_offset-1).to_float64().to_int()
        var rand_y = pyrandom.randint(-game_height_offset, game_height_offset-1).to_float64().to_int()
        self.food_array.append(Vector2D(rand_x, rand_y))

    fn update_habitat(inout self) raises:
        var pygame = Python.import_module("pygame")

        self.screen.fill((0, 0, 0))

        # Check if all snakes are dead after update cycle
        self.active = False
        for index in range(snake_count):
            # Resize food array if necessary
            if self.habitat[index].score - starting_score >= len(self.food_array):
                self.generate_food()
            
            # Update snake
            var fruit_position = self.food_array[self.habitat[index].score - starting_score]
            self.habitat[index].update(self.screen, fruit_position, self.stats)

            # Check if any other snakes are still active
            if self.active:
                continue
        
            # Check if this snake is dead
            if not self.habitat[index].is_dead():
                self.active = True
            
        self.draw_latest_food()
        pygame.display.update()

    fn generate_next_habitat(inout self) raises:
        var max_fitness_value: Int
        var max_fitness_index: Int
        var calculated_average: Float64
        var previous_stats = self.stats
        var parent_traits = NeuralNetwork[SPEC]()
        max_fitness_value, max_fitness_index, calculated_average = self.set_max_fitness()
        self.set_best_snake(max_fitness_value, max_fitness_index)
        self.update_stats(
            generation=self.stats["generation"] + 1,
            max=max_fitness_value,
            average=calculated_average,
            best=self.best_snake.snake.fitness
        )
        self.log_stats(previous_stats)
        parent_traits.copy(self.best_snake.snake.neural_network)
        self.mutate_population(max_fitness_index, parent_traits)
        self.reset_population()

    fn reset_population(inout self):
        for i in range(snake_count):
            self.habitat[i].reset()

        if len(self.food_array) > 1:
            self.food_array = List(self.food_array[-1])

        self.active = True

    fn set_max_fitness(inout self) -> (Int, Int, Float64):
        var max_fitness_value = 0
        var max_fitness_index = 0
        var fitness_sum = 0
        for index in range(snake_count):
            var fitness = self.habitat[index].fitness
            if self.habitat[index].fitness > max_fitness_value:
                max_fitness_value = fitness
                max_fitness_index = index
            fitness_sum += fitness

        return max_fitness_value, max_fitness_index, fitness_sum / snake_count

    fn set_best_snake(inout self, new_value: Int, new_index: Int) raises:
        var previous_stats = self.stats
        if new_value > int(previous_stats["best"]):
            self.save()
            var snake = Snake()
            snake.fitness = new_value
            snake.neural_network.copy(self.habitat[new_index].neural_network)
            self.stats["best"] = new_value
            self.best_snake = SnakeHandler[snake_count, mutation_rate](snake, self.food_array, self.stats, self.screen)
            self.log_stats(previous_stats)
            self.logger.notice("New best snake found with fitness " + str(new_value) + "! Replay requested")
            self.replay_active = True
        else:
            self.habitat[new_index].neural_network.copy(self.best_snake.snake.neural_network)

        if self.replay_active:
            self.logger.notice("Replay started")
            self.best_snake.replay(0.1)
            self.best_snake.snake.fitness = int(self.stats["best"])
            self.replay_active = False

        

    fn mutate_population(inout self, max_index: Int, parent_traits: NeuralNetwork[SPEC]) raises:
        for index in range(snake_count):
            if index != max_index:
                self.habitat[index].neural_network.copy(parent_traits)
                self.habitat[index].neural_network.mutate(mutation_rate)

    fn update_stats(inout self, **kwargs: Float32) raises:
        for key in kwargs.keys():
            var current_key = key[]
            if current_key in self.stats:
                self.stats[current_key] = kwargs[current_key]
    
    fn log_stats(inout self, previous_stats: PopulationStats) raises:
        self.logger.cls()
        for key in self.stats.keys():
            var stat = key[]
            if self.stats[stat] <= previous_stats[stat]:
                self.logger.status(str(stat) + ": " + str(self.stats[stat]) + " (-)")
                continue
                
            self.logger.status(str(stat) + ": " + str(self.stats[stat]) + " (+)")

    fn draw_latest_food(inout self) raises:
        Self.draw_food(self.screen, self.food_array[-1], (0, 200, 0))

    fn draw_all_food(inout self, food_array: List[Vector2D]) raises:
        Self.draw_food(self.screen, food_array[-1], (0, 200, 0))
        for index in range(len(food_array) - 1):
            Self.draw_food(self.screen, food_array[index], (75, 75, 75))

    fn draw_all_food(inout self) raises:
        self.draw_all_food(self.food_array)

    @staticmethod
    fn draw_food(screen: PythonObject, position: Vector2D, color: RGB) raises:
        var pygame = Python.import_module("pygame")
        var food_x = position[0] + game_width_offset
        var food_y = position[1] + game_height_offset
        pygame.draw.rect(screen, color, (int(food_x) * game_scale, int(food_y) * game_scale, game_scale, game_scale))

    fn save(inout self) raises:
        self.best_snake.save()
        var neural_network_filename_prefix = "data/" + str(self.habitat[0].neural_network.__repr__())
        var checkpoint_filename_prefix = "checkpoints/" + str(self.habitat[0].neural_network.__repr__())
        self.logger.notice("NeuralNetwork data serialized at " + neural_network_filename_prefix + "-#")
        self.logger.notice("SnakeHandler data serialized at " + checkpoint_filename_prefix)

    fn wait_for_save(inout self) raises:
        var input = Python.import_module("builtins").input
        while True:
            var save_population = input("Save population data?(Y/n)")
            if save_population.lower()[0] == "n":
                return
            elif save_population.lower()[0] == "y":
                break
        self.save()

    fn load(inout self, mutation_rate: Float32) raises:
        for habitat_index in range(1, snake_count):
            self.habitat[habitat_index].neural_network.copy(self.best_snake.snake.neural_network)
            self.habitat[habitat_index].neural_network.mutate(mutation_rate)

        var filename_prefix = "data/" + str(self.habitat[0].neural_network.__repr__())
        self.logger.notice("NeuralNetwork data deserialized from " + filename_prefix + "-#")