from population import Population, Vector2D, PopulationStats, RGB, starting_score, game_width, game_height, game_width_offset, game_height_offset, game_scale, dtype
from snake import Snake
from python import Python
from time import sleep
from tensor import Tensor
from logger import Logger

struct SnakeHandler[snake_count: Int, mutation_rate: Float32]:
    var stats: PopulationStats
    var snake: Snake
    var food_array: List[Vector2D]
    var screen: PythonObject

    fn __init__(inout self, snake: Snake, food_array: List[Vector2D], stats: PopulationStats, screen: PythonObject) raises:
        self.stats = stats
        self.snake = Snake()
        self.snake.neural_network.copy(snake.neural_network)
        self.snake.fitness = snake.fitness
        self.food_array = food_array
        self.screen = screen

    fn replay(inout self, replay_speed: Float64) raises:
        self.snake.reset()
        var current_speed = replay_speed
        var pygame = Python.import_module("pygame")
        var active_food = self.food_array[0]
        var clock = pygame.time.Clock()
        var run = True
        while not self.snake.is_dead() and run:
            var events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    run = False
            
            var keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                current_speed = 0
            elif keys[pygame.K_DOWN]:
                current_speed = replay_speed * 4
            else:
                current_speed = replay_speed
                
            self.screen.fill((0, 0, 0))
            var net_score = self.snake.score - starting_score
            self.snake.update(self.screen, self.food_array[net_score], self.stats)
            self.draw_all_food(self.food_array[0:net_score+1])
            self.draw_fitness()
            pygame.display.update()
            sleep(current_speed)

    fn reset(inout self):
        var preserved_fitness = self.snake.fitness
        self.snake.reset()
        self.snake.fitness = preserved_fitness

    fn draw_all_food(inout self, food_array: List[Vector2D]) raises:
        Self.draw_food(self.screen, food_array[-1], (0, 200, 0))
        for index in range(len(food_array) - 1):
            Self.draw_food(self.screen, food_array[index], (75, 75, 75))

    @staticmethod
    fn draw_food(screen: PythonObject, position: Vector2D, color: RGB) raises:
        var pygame = Python.import_module("pygame")
        var food_x = position[0] + game_width_offset
        var food_y = position[1] + game_height_offset
        pygame.draw.rect(screen, color, (int(food_x) * game_scale, int(food_y) * game_scale, game_scale, game_scale))

    fn draw_fitness(inout self) raises:
        var pygame = Python.import_module("pygame")
        var font = pygame.font.Font(None, 36)
        var text = font.render("Fitness: " + str(self.snake.fitness), True, (255, 255, 255))
        self.screen.blit(text, (game_scale, game_scale))


    fn save(inout self) raises:
        var tensor = Tensor[dtype](len(self.food_array), 2)
        for index in range(len(self.food_array)):
            var index_x = StaticIntTuple[2](index, 0)
            var index_y = StaticIntTuple[2](index, 1)
            var value_x = self.food_array[index][0]
            var value_y = self.food_array[index][1]
            tensor.__setitem__(index_x, value_x)
            tensor.__setitem__(index_y, value_y)

        var filename = "checkpoints/" + self.snake.neural_network.__repr__()
        tensor.tofile(filename)
        self.snake.neural_network.save()

    fn load(inout self) raises:
        var filename = "checkpoints/" + self.snake.neural_network.__repr__()
        var tensor = Tensor[dtype].fromfile(filename)
        var food_count = tensor.shape()[0]
        for index in range(0, food_count, 2):
            var index_x = StaticIntTuple[1](index)
            var index_y = StaticIntTuple[1](index+1)
            var value_x = tensor.__getitem__(index_x)
            var value_y = tensor.__getitem__(index_y)
            self.food_array.append(Vector2D(value_x, value_y))
        self.snake.neural_network.load()