from python import Python
from population import Population, game_width, game_height, game_scale
from logger import Logger
from time import sleep

alias snake_count: Int = 30

fn main() raises:
    Logger.notice("Starting simulation of " + str(snake_count) + " snakes...")
    print("pre")
    var population = Population[snake_count]()
    var pygame = Python.import_module("pygame")
    var screen = pygame.display.set_mode((game_width * game_scale, game_height * game_scale))
    var run = True
    while run:
        var events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                run = False
        var timeout = 300
        while population.update_habitat(screen) and timeout > 0:
            Logger.notice("Updating generation...")
            timeout -= 1
        Logger.notice("Generation has died. Generating next habitat...")
        population.generate_next_habitat(survival_rate=0.5)
        sleep(0.5)
        
    