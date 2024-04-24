from python import Python
from population import Population, game_width, game_height, game_scale
from logger import Logger
from time import sleep

alias snake_count: Int = 50
alias timeout: Int = 200 # Number of snakes steps before population is automatically updated

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
            survival_rate=0.25,
            mutation_rate=1.2
        )