from python import Python
from population import Population, game_width, game_height, game_scale
from logger import Logger
from time import sleep

alias snake_count: Int = 50
alias timeout: Int = 200 # Number of snakes steps before population is automatically updated

fn main() raises:
    var pygame = Python.import_module("pygame")
    var input = Python.import_module("builtins").input
    
    var population = Population[snake_count]()
    Logger.cls()
    try:
        population.load()
    except:
        Logger.warn("No serialized data found. Starting new population.")

    var run = True
    while run:
        var count = 0
        while count < timeout and population.active and run:
            var events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    run = False

            population.update_habitat()
            count += 1
        population.generate_next_habitat(survival_rate=0.5)

    while True:
        var save_population = input("Save population data?(Y/n)")
        if save_population.lower()[0] == "n":
            return
        elif save_population.lower()[0] == "y":
            break

    try:
        population.save()
    except:
        Logger.error("Population save unsuccessful. FeelsBadMan.")