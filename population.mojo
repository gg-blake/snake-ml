from python import Python
from snake import Snake
from neural_network import NeuralNetwork, NeuralNetworkSpec, NeuralNetworkShape
from math import sqrt, abs, clamp
from tensor import Tensor
from collections.vector import InlinedFixedVector
from logger import Logger
from time import sleep
from buffer import Buffer
from snake_handler import SnakeHandler

# Screen constants
alias GAME_WIDTH: Int = 40
alias GAME_HEIGHT: Int = 40
alias GAME_DEPTH: Int = 40
alias GAME_WIDTH_OFFSET: Int = GAME_WIDTH // 2
alias GAME_HEIGHT_OFFSET: Int = GAME_HEIGHT // 2
alias GAME_DEPTH_OFFSET: Int = GAME_DEPTH // 2
alias GAME_SCALE: Int = 26

# Neural network constants
alias DTYPE = DType.float32
alias SHAPE = NeuralNetworkShape(15, 30, 30, 5)
alias SPEC = NeuralNetworkSpec(DTYPE, SHAPE)

# Population constants
alias INITIAL_SCORE: Int = 5
alias TTL: Vector1D = 100
alias SNAKE_COUNT: Int = 35
alias MUTATION_RATE: Float32 = 1

# Type aliases
alias Vector3D = SIMD[DTYPE, 4]
alias Vector1D = SIMD[DTYPE, 1]
alias PopulationStats = Dict[String, Scalar[DTYPE]]
alias RGB = Tuple[Int, Int, Int]

struct Population:
    var screen: PythonObject
    var logger: Logger
    var habitat: AnyPointer[Snake]
    var food_array: List[Vector3D]
    var stats: PopulationStats
    var best_snake: SnakeHandler
    var active: Bool
    var replay_active: Bool
    
    fn __init__(inout self) raises:
        var pygame = Python.import_module("pygame")
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Snake AI")
        self.screen = pygame.display.set_mode((GAME_WIDTH * GAME_SCALE, GAME_HEIGHT * GAME_SCALE))
        self.logger = Logger("logs")
        self.habitat = AnyPointer[Snake].alloc(SNAKE_COUNT)
        self.food_array = List[Vector3D]()
        self.stats = PopulationStats()
        self.stats["generation"] = 0
        self.stats["max"] = TTL
        self.stats["average"] = 0
        self.stats["best_fitness"] = 0
        self.stats["best_score"] = 0
        self.best_snake = SnakeHandler(Snake(), self.food_array, self.stats, self.screen)
        self.active = True
        self.replay_active = False
        self.init_habitat()
        

    fn __getitem__(inout self, idx: Int) raises -> NeuralNetwork[SPEC]:
        if idx >= SNAKE_COUNT or idx < 0:
            raise Error("Habitat index out of range")

        return self.habitat[idx].neural_network

    fn __del__(owned self):
        self.habitat.free()
        try:
            self.wait_for_save()
        except:
            self.logger.error("Population save unsuccessful. FeelsBadMan.")

    fn reset_stats(inout self) raises:
        self.stats["generation"] = 0
        self.stats["max"] = TTL
        self.stats["average"] = 0
        self.stats["best_fitness"] = 0
        self.stats["best_score"] = 0

    fn init_habitat(inout self) raises:
        self.generate_food()
        for id in range(SNAKE_COUNT):
            self.habitat[id] = Snake()
        try:
            self.best_snake.load()
            self.log_stats(self.stats)
            self.best_snake.replay(0.1)
            self.stats["best_fitness"] = self.best_snake.snake.fitness
            self.load()
            
            
        except:
            self.logger.warn("No serialized data found. Starting new population.")

    fn generate_food(inout self) raises:
        var pyrandom = Python.import_module("random")
        var rand_x = pyrandom.randint(-GAME_WIDTH_OFFSET, GAME_WIDTH_OFFSET-1).to_float64().to_int()
        var rand_y = pyrandom.randint(-GAME_HEIGHT_OFFSET, GAME_HEIGHT_OFFSET-1).to_float64().to_int()
        self.food_array.append(Vector3D(rand_x, rand_y))

    fn update_habitat(inout self) raises:
        var pygame = Python.import_module("pygame")

        self.screen.fill((0, 0, 0))

        # Check if all snakes are dead after update cycle
        self.active = False
        for index in range(SNAKE_COUNT):
            # Resize food array if necessary
            if self.habitat[index].score - INITIAL_SCORE >= len(self.food_array):
                self.generate_food()
            
            # Update snake
            var fruit_position = self.food_array[self.habitat[index].score - INITIAL_SCORE]
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
        var max_fitness_value: Vector1D
        var max_fitness_index: Int
        var calculated_average: Vector1D
        var previous_stats = self.stats
        var parent_traits = NeuralNetwork[SPEC]()
        max_fitness_value, max_fitness_index, calculated_average = self.set_max_fitness()
        self.set_best_snake(max_fitness_value, max_fitness_index)
        self.update_stats(
            generation=self.stats["generation"] + 1,
            max=max_fitness_value,
            average=calculated_average,
            best_fitness=self.best_snake.snake.fitness,
            best_score=self.best_snake.snake.score-INITIAL_SCORE
        )
        self.log_stats(previous_stats)
        parent_traits.copy(self.best_snake.snake.neural_network)
        self.mutate_population(max_fitness_index, parent_traits)
        self.reset_population()

    fn reset_population(inout self):
        for i in range(SNAKE_COUNT):
            self.habitat[i].reset()

        if len(self.food_array) > 1:
            self.food_array = List(self.food_array[-1])

        self.active = True

    fn set_max_fitness(inout self) -> (Vector1D, Int, Vector1D):
        var max_fitness_value: Vector1D = 0
        var max_fitness_index: Int = 0
        var fitness_sum: Vector1D = 0
        for index in range(SNAKE_COUNT):
            var fitness: Vector1D = self.habitat[index].fitness
            if self.habitat[index].fitness > max_fitness_value:
                max_fitness_value = fitness
                max_fitness_index = index
            fitness_sum += fitness

        return max_fitness_value, max_fitness_index, fitness_sum / SNAKE_COUNT

    fn set_best_snake(inout self, new_value: Vector1D, new_index: Int) raises:
        var previous_stats = self.stats
        if new_value > int(previous_stats["best_fitness"]):
            self.save()
            var snake = Snake()
            snake.fitness = new_value
            snake.neural_network.copy(self.habitat[new_index].neural_network)
            self.stats["best_fitness"] = new_value
            self.best_snake = SnakeHandler(snake, self.food_array, self.stats, self.screen)
            self.log_stats(previous_stats)
            self.logger.notice("New best snake found with fitness " + str(new_value) + "! Replay requested")
            self.replay_active = True
        else:
            self.habitat[new_index].neural_network.copy(self.best_snake.snake.neural_network)

        if self.replay_active:
            self.logger.notice("Replay started")
            self.best_snake.replay(0.1)
            self.best_snake.snake.fitness = int(self.stats["best_fitness"])
            self.replay_active = False

        

    fn mutate_population(inout self, max_index: Int, parent_traits: NeuralNetwork[SPEC]) raises:
        for index in range(SNAKE_COUNT):
            if index != max_index:
                self.habitat[index].neural_network.copy(parent_traits)
                self.habitat[index].neural_network.mutate(MUTATION_RATE)

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

    fn draw_all_food(inout self, food_array: List[Vector3D]) raises:
        Self.draw_food(self.screen, food_array[-1], (0, 200, 0))
        for index in range(len(food_array) - 1):
            Self.draw_food(self.screen, food_array[index], (75, 75, 75))

    fn draw_all_food(inout self) raises:
        self.draw_all_food(self.food_array)

    @staticmethod
    fn draw_food(screen: PythonObject, position: Vector3D, color: RGB) raises:
        var pygame = Python.import_module("pygame")
        var food_x = position[0] + GAME_WIDTH_OFFSET
        var food_y = position[1] + GAME_HEIGHT_OFFSET
        var food_z = position[2] + GAME_DEPTH_OFFSET
        var food_z_scaled = int(clamp(food_z / GAME_DEPTH, 0, 1))
        var weighted_color = (color.get[0, Int]() * food_z_scaled, color.get[1, Int]() * food_z_scaled, color.get[2, Int]() * food_z_scaled)
        pygame.draw.rect(screen, weighted_color, (int(food_x) * GAME_SCALE, int(food_y) * GAME_SCALE, GAME_SCALE, GAME_SCALE))

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

    fn load(inout self) raises:
        for habitat_index in range(1, SNAKE_COUNT):
            self.habitat[habitat_index].neural_network.copy(self.best_snake.snake.neural_network)
            self.habitat[habitat_index].neural_network.mutate(MUTATION_RATE)

        var filename_prefix = "data/" + str(self.habitat[0].neural_network.__repr__())
        self.logger.notice("NeuralNetwork data deserialized from " + filename_prefix + "-#")