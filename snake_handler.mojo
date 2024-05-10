from population import Population, Vec3, Vec1, PopulationStats, RGB, INITIAL_SCORE, GAME_SCALE, GAME_HEIGHT_OFFSET, GAME_WIDTH_OFFSET, GAME_DEPTH_OFFSET, GAME_DEPTH, DTYPE
from snake import Snake
from python import Python
from time import sleep
from tensor import Tensor, TensorShape
from logger import Logger
from math import clamp
from game_object import GameObject3D
from utils.index import Index

struct SnakeHandler:
    var stats: PopulationStats
    var snake: Snake
    var food_array: List[GameObject3D]
    var screen: PythonObject

    fn __init__(inout self, snake: Snake, food_array: List[GameObject3D], stats: PopulationStats, screen: PythonObject) raises:
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
            var net_score = self.snake.score - INITIAL_SCORE
            self.snake.update(self.screen, self.food_array[net_score], self.stats)
            self.draw_all_food(self.food_array[0:net_score+1])
            self.draw_fitness()
            pygame.display.update()
            sleep(current_speed)

    fn reset(inout self) raises:
        var preserved_fitness = self.snake.fitness
        self.snake.reset()
        self.snake.fitness = preserved_fitness

    fn draw_all_food(inout self, food_array: List[GameObject3D]) raises:
        Self.draw_food(self.screen, food_array[-1], (0, 200, 0))
        for index in range(len(food_array) - 1):
            Self.draw_food(self.screen, food_array[index], (75, 75, 75))

    @staticmethod
    fn draw_food(screen: PythonObject, position: GameObject3D, color: RGB) raises:
        var pygame = Python.import_module("pygame")
        var food_x = position[0] + GAME_WIDTH_OFFSET
        var food_y = position[1] + GAME_HEIGHT_OFFSET
        var food_z = position[2] + GAME_DEPTH_OFFSET
        var food_z_scaled = int(clamp(food_z / GAME_DEPTH, 0, 1))
        var weighted_color = (color.get[0, Int]() * food_z_scaled, color.get[1, Int]() * food_z_scaled, color.get[2, Int]() * food_z_scaled)
        pygame.draw.rect(screen, weighted_color, (int(food_x) * GAME_SCALE, int(food_y) * GAME_SCALE, GAME_SCALE, GAME_SCALE))

    fn draw_fitness(inout self) raises:
        var pygame = Python.import_module("pygame")
        var font = pygame.font.Font(None, 36)
        var text = font.render("Fitness: " + str(self.snake.fitness), True, (255, 255, 255))
        self.screen.blit(text, (GAME_SCALE, GAME_SCALE))


    fn save(inout self) raises:
        var tensor = Tensor[DTYPE](TensorShape(len(self.food_array), 3))
        for vec_i in range(len(self.food_array)):
            for vec_j in range(3):
                tensor[Index(vec_i, vec_j)] = self.food_array[vec_i].position[vec_j]

        var filename = "checkpoints/" + self.snake.neural_network.__repr__()
        tensor.tofile(filename)
        self.snake.neural_network.save()

    fn load(inout self) raises:
        var filename = "checkpoints/" + self.snake.neural_network.__repr__()
        var tensor = Tensor[DTYPE].fromfile(filename)
        var food_count = tensor.shape()[0]
        var vec_size = tensor.shape()[1]
        for vec_i in range(food_count):
            var positions = List[Vec1]()
            for vec_j in range(vec_size):
                positions.append(tensor[Index(vec_i, vec_j)])
            
            var x = positions[0]
            var y = positions[1]
            var z = positions[2]
            self.food_array.append(GameObject3D(x, y, z))
        self.snake.neural_network.load()