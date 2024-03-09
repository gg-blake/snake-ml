from python import Python
from population import Population
from logger import Logger
from time import sleep

alias snake_count: Int = 100

fn main() raises:
    Logger.cls()
    Logger.notice("Starting simulation of " + str(snake_count) + " snakes...")
    var population = Population[snake_count]()
    sleep(1)
    while True:
        Logger.cls()
        while population.habitat_status():
            population.update_habitat()
            Logger.notice("Current Generation: " + str(population.generation))
            Logger.notice("Simulating population...")
        Logger.notice("Generation has died. Generating next habitat...")
        population.generate_next_habitat(survival_rate=0.5)
        sleep(0.5)