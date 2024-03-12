from python import Python
from population import Population
from logger import Logger
from time import sleep

alias snake_count: Int = 100

fn main() raises:
    Logger.notice("Starting simulation of " + str(snake_count) + " snakes...")
    print("pre")
    var population = Population[snake_count]()
    print("done")
    while True:
        while population.update_habitat():
            Logger.notice("Running simulation...")
        Logger.notice("Generation has died. Generating next habitat...")
        population.generate_next_habitat(survival_rate=0.5)
        sleep(0.5)