from population import Vector2D, PopulationStats, RGB, starting_score, game_width, game_height, game_width_offset, game_height_offset, game_scale, dtype
from snake import Snake
from python import Python
from time import sleep
from buffer import Buffer

struct PopulationPlayback[snake_count: Int, mutation_rate: Float32]:
    var stats: PopulationStats
    var snake: Snake
    var food_array: List[Vector2D]
    var screen: PythonObject

    fn __init__(inout self, snake: Snake, food_array: List[Vector2D], stats: PopulationStats, screen: PythonObject) raises:
        self.stats = stats
        self.snake = Snake()
        self.snake.neural_network.copy(snake.neural_network)
        self.food_array = food_array
        self.screen = screen

    fn replay(inout self, replay_speed: Float64) raises:
        var pygame = Python.import_module("pygame")
        var active_food = self.food_array[0]
        var clock = pygame.time.Clock()
        var run = True
        while not self.snake.is_dead() and run:
            var events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    run = False
            self.screen.fill((0, 0, 0))
            var net_score = self.snake.score - starting_score
            self.snake.update(self.screen, self.food_array[net_score], self.stats)
            self.draw_all_food(self.food_array[0:net_score+1])
            pygame.display.update()
            sleep(replay_speed)

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

    '''fn save(inout self) raises:
        alias max_food = game_width * game_height * 2
        var buf = Buffer[dtype, max_food]()
        for i in range(len(self.food_array)):
            var position = self.food_array[i]
            buf.store[width=2](i, position)'''