import numpy as np
import pygame, os, json, sys, time
from .constants import *
from .population import Population
from .snake_obj import Snake
from .food_obj import Food
from .snake import screen
import matplotlib.pyplot as plt


# Intialize pygame window for plotting average distance graphs
pygame.init()
pygame.display.set_caption("Snake AI")
graph_screen = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))

clock = pygame.time.Clock()
# Set record mode
write_interval = 1
write_location = ""
write_mode = False
try:
    if sys.argv[1] == "write":
        write_mode = True
        write_location = sys.argv[2]
        write_interval = int(sys.argv[3])
    elif sys.argv[1] == "read":
        write_location = sys.argv[2]
except IndexError:
    print("No arguments given, running in normal mode")

os.system("cls")
print(f"Write mode: {write_mode}\nWrite interval: every {write_interval} generations\nReading/Write at: ./{write_location}")
time.sleep(2)
# Create the population
population = Population(100, 12, 16, 4, 0.5)
# Load game state and preserve progress

if os.path.exists(write_location):
    print("Save file found, loading population")
    with open(write_location, "r") as f:
        data = json.load(f)

        population.generation = data["generation"]
        for i, v in enumerate(data["population"]):
            population.population[i].nn.input_nodes = v['input_nodes']
            population.population[i].nn.hidden_nodes = v['hidden_nodes']
            population.population[i].nn.output_nodes = v['output_nodes']
            population.population[i].nn.weights_ih = np.array(v['weights_ih'])
            population.population[i].nn.weights_ho = np.array(v['weights_ho'])
            population.population[i].nn.bias_h = np.array(v['bias_h'])
            population.population[i].nn.bias_o = np.array(v['bias_o'])
else:
    print("No save file found, creating new population")
initial_generation = population.generation
time.sleep(2)
# Create the snakes
snakes = [population.get_random() for _ in range(population.size)]

stat_distance = []
stat_fitness = []

# Main loop
while True:
    try:
        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if population.generation % write_interval == 0 and population.generation > 0 and write_mode:
            # Save population.population to json
            with open(write_location, "w") as f:
                json.dump(population.__dict__(), f)

        

        # Check if all the snakes are dead
        if all([snake.dead for snake in snakes]):
            # Store old PR
            old_best = population.best
            # Calculate the fitness of each snake
            for snake in snakes:
                snake.calculate_fitness()
            
            # Ensure the next generational best is an improvement on last generation
            if population.best.fitness < old_best.fitness:
                population.population.append(old_best)
                population.best = old_best
            #population.fitness_decay()
            # Plot the fitness of the best snake
            stat_distance.append(population.get_average_distance())
            if len(stat_distance) > 100:
                stat_distance.pop(0)
            # Natural selection
            population.natural_selection()
            # Create new snakes
            snakes = [population.get_random() for _ in range(population.size)]
            Food._count = 0
            Snake._active_food = [Food(Snake._active_food[-1].x, Snake._active_food[-1].y)]
            

        # Loop through all the snakes
        for snake in snakes:
            # If the snake is not dead
            if not snake.dead:
                # Update
                snake.update()

        # Draw the background
        screen.fill((0, 0, 0))
        for food in Snake._active_food:
            food.draw()
        # Loop through all the snakes
        for snake in snakes:
            # If the snake is not dead
            if not snake.dead:
                # Draw the snake
                snake.draw()

        # Print Population Stats
        os.system("cls")
        print(population)
        # Update the screen
        pygame.display.update()
        # Set the frame rate
        clock.tick(60)
    except KeyboardInterrupt:
        break

# Plot the avg points
plt.plot([*range(initial_generation, population.generation)], stat_distance)
plt.xlim(initial_generation, population.generation)
plt.show()