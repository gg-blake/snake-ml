from python import Python
from population import Population, game_width, game_height, game_scale
from logger import Logger
from time import sleep

alias snake_count: Int = 200 # Keep this 30 and under to reduce risk of segmentation error when adding `Position` to `self.body_set`

fn main() raises:
    Logger.notice("Starting simulation of " + str(snake_count) + " snakes...")
    var pygame = Python.import_module("pygame")
    var screen = pygame.display.set_mode((game_width * game_scale, game_height * game_scale))
    var population = Population[snake_count]()
    
    var run = True
    while run:
        var count = 0
        while population.update_habitat(screen) and count < 400 and run:
            var events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    run = False
            #count += 1
            Logger.notice("Updating habitat")
        Logger.notice("Generation has died. Generating next habitat...")
        population.generate_next_habitat(survival_rate=0.2)
        
    