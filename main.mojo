from python import Python
from population import Population
from logger import Logger
from time import sleep

fn main() raises:
    var pygame = Python.import_module("pygame")
    var population = Population()
    var run = True
    while run:
        while population.active and run:
            var events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    run = False
            
            var keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE] and not population.replay_active:
                population.replay_active = True
                population.logger.notice("Replay requested")

            population.update_habitat()
        population.generate_next_habitat()