# Create a neural network that learns to play snake
import numpy as np
import random
import math
import pygame
import time 

WIDTH = 10
HEIGHT = 10
SCALE = 50


# Create the snake
class Snake:
    def __init__(self, x, y, nn):
        self.x = x
        self.y = y
        self.nn = nn
        self.fitness = 0
        self.direction = 0
        self.history = []
        self.dead = False
        self.food_eaten = 0
        self.score = 0
        self.moves = 0

    def think(self, food):
        inputs = np.array([self.x, self.y, food.x, food.y])
        output = self.nn.feed_forward(inputs)
        self.direction = np.argmax(output)

    def update(self, food):
        self.moves += 1
        self.history.append((self.x, self.y))

        if self.direction == 0:
            self.x += 1
        elif self.direction == 1:
            self.y += 1
        elif self.direction == 2:
            self.x -= 1
        elif self.direction == 3:
            self.y -= 1

        if self.x > WIDTH - 1:
            self.x = 0
        elif self.x < 0:
            self.x = WIDTH - 1

        if self.y > HEIGHT - 1:
            self.y = 0
        elif self.y < 0:
            self.y = HEIGHT - 1

        if self.x == food.x and self.y == food.y:
            self.food_eaten += 1
            self.score += 1
            return True
        else:
            self.history.pop(0)
            return False

    def calculate_fitness(self):
        self.fitness = self.score * self.score * 100 + self.food_eaten * 1000 - self.moves

    def is_dead(self):
        if (self.x, self.y) in self.history[:-1]:
            return True
        else:
            return False

    def copy(self):
        snake = Snake(self.x, self.y, self.nn.copy())
        snake.fitness = self.fitness
        snake.direction = self.direction
        snake.history = self.history.copy()
        snake.dead = self.dead
        snake.food_eaten = self.food_eaten
        snake.score = self.score
        snake.moves = self.moves
        return snake

    def __str__(self):
        return f"Score: {self.score}, Fitness: {self.fitness}"
    
    def __repr__(self):
        return self.__str__()

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        # Number of nodes in each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Weights
        self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes)
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes)

        # Biases
        self.bias_h = np.random.rand(self.hidden_nodes, 1)
        self.bias_o = np.random.rand(self.output_nodes, 1)

        self.fitness = 0

    def feedforward(self, input_array):
        inputs = np.array(input_array, ndmin=2).T

        # Dot multiply the weights with the inputs
        hidden = np.dot(self.weights_ih, inputs)
        hidden = np.add(hidden, self.bias_h)
        hidden = self.sigmoid(hidden)

        output = np.dot(self.weights_ho, hidden)
        output = np.add(output, self.bias_o)
        output = self.sigmoid(output)

        return output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, input_array, target_array):
        inputs = np.array(input_array, ndmin=2).T
        targets = np.array(target_array, ndmin=2).T

        hidden = np.dot(self.weights_ih, inputs)
        hidden = np.add(hidden, self.bias_h)
        hidden = self.sigmoid(hidden)

        output = np.dot(self.weights_ho, hidden)
        output = np.add(output, self.bias_o)
        output = self.sigmoid(output)

        output_errors = np.subtract(targets, output)
        hidden_errors = np.dot(self.weights_ho.T, output_errors)

        self.weights_ho = np.add(self.weights_ho, np.dot((output_errors * self.sigmoid_derivative(output)), hidden.T))
        self.weights_ih = np.add(self.weights_ih, np.dot((hidden_errors * self.sigmoid_derivative(hidden)), inputs.T))

        self.bias_o = np.add(self.bias_o, output_errors * self.sigmoid_derivative(output))
        self.bias_h = np.add(self.bias_h, hidden_errors * self.sigmoid_derivative(hidden))

    def mutate(self, mutation_rate):
        def mutate(val):
            if random.random() < mutation_rate:
                return val + random.gauss(0, 0.1)
            else:
                return val

        mutate = np.vectorize(mutate)

        self.weights_ih = mutate(self.weights_ih)
        self.weights_ho = mutate(self.weights_ho)
        self.bias_h = mutate(self.bias_h)
        self.bias_o = mutate(self.bias_o)
        
    def copy(self):
        nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        nn.weights_ih = np.copy(self.weights_ih)
        nn.weights_ho = np.copy(self.weights_ho)
        nn.bias_h = np.copy(self.bias_h)
        nn.bias_o = np.copy(self.bias_o)
        return nn
    
    def __str__(self):
        return f"weights_ih: {self.weights_ih}, weights_ho: {self.weights_ho}, bias_h: {self.bias_h}, bias_o: {self.bias_o}"
    
    def __repr__(self):
        return self.__str__()
    
# Create the population
class Population:
    def __init__(self, size, input_nodes, hidden_nodes, output_nodes):
        self.size = size
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.generation = 0
        self.mutation_rate = 0.1
        self.best = None
        self.best_fitness = 0

        self.population = []
        for i in range(self.size):
            self.population.append(NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes))

    def get_best(self):
        return self.best
    
    def get_best_fitness(self):
        return self.best_fitness
    
    def get_generation(self):
        return self.generation
    
    def get_average_fitness(self):
        return sum([snake.fitness for snake in self.population]) / len(self.population)
    
    def get_worst_fitness(self):
        return min([snake.fitness for snake in self.population])
    
    def get_worst(self):
        return min(self.population, key=lambda snake: snake.fitness)
    
    def get_random(self):
        return random.choice(self.population)
    
    def get_fittest(self):
        return max(self.population, key=lambda snake: snake.fitness)
    
    def get_fittest_fitness(self):
        return max([snake.fitness for snake in self.population])
    
    def get_fittest_index(self):
        return self.population.index(self.get_fittest())
    
    def get_worst_index(self):
        return self.population.index(self.get_worst())
    
    def crossover(self, parent_a, parent_b):
        child = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        child.weights_ih = np.add(parent_a.weights_ih, parent_b.weights_ih) / 2
        child.weights_ho = np.add(parent_a.weights_ho, parent_b.weights_ho) / 2
        child.bias_h = np.add(parent_a.bias_h, parent_b.bias_h) / 2
        child.bias_o = np.add(parent_a.bias_o, parent_b.bias_o) / 2
        return child

    def natural_selection(self):
        new_population = []

        # Elitism
        self.best = self.get_fittest()
        self.best_fitness = self.get_fittest_fitness()
        new_population.append(self.best.copy())

        # Crossover
        for i in range(self.size - 1):
            parent_a = self.select_parent()
            parent_b = self.select_parent()
            child = self.crossover(parent_a, parent_b)
            child.mutate(self.mutation_rate)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def select_parent(self):
        index = 0
        r = random.uniform(0, self.get_fittest_fitness())
        

        while r > 0:
            r = r - self.population[index].fitness
            index += 1

        index -= 1
        return self.population[index]
    
    def __str__(self):
        return f"Generation: {self.generation}, Best Fitness: {self.best_fitness}"
    
    def __repr__(self):
        return self.__str__()


# Create the snake
class Snake:
    def __init__(self, x, y, nn):
        self.x = x
        self.y = y
        self.nn = nn
        self.fitness = 0
        self.direction = 0
        self.history = []
        self.dead = False
        self.food_eaten = 0
        self.score = 0
        self.moves = 0
        self.movesnofood = 0

    def think(self, food):
        inputs = np.array([self.x, self.y, food.x, food.y])
        output = self.nn.feedforward(inputs)
        self.direction = np.argmax(output)

    def update(self, food):
        if self.movesnofood > 100:
            self.dead = True

        self.moves += 1
        self.movesnofood += 1
        self.history.append((self.x, self.y))

        if self.direction == 0:
            self.x += 1
        elif self.direction == 1:
            self.y += 1
        elif self.direction == 2:
            self.x -= 1
        elif self.direction == 3:
            self.y -= 1

        if self.x > WIDTH - 1:
            self.x = 0
        elif self.x < 0:
            self.x = WIDTH - 1

        if self.y > HEIGHT - 1:
            self.y = 0
        elif self.y < 0:
            self.y = HEIGHT - 1

        if self.x == food.x and self.y == food.y:
            self.food_eaten += 1
            self.score += 1
            self.movesnofood = 0
            return True
        else:
            self.history.pop(0)
            return False
        
        

    def calculate_fitness(self):
        self.fitness = self.score * self.score * 100 + self.food_eaten * 1000 - self.moves

    def is_dead(self):
        if (self.x, self.y) in self.history[:-1]:
            return True
        else:
            return False

    def copy(self):
        snake = Snake(self.x, self.y, self.nn.copy())
        snake.fitness = self.fitness
        snake.direction = self.direction
        snake.history = self.history.copy()
        snake.dead = self.dead
        snake.food_eaten = self.food_eaten
        snake.score = self.score
        snake.moves = self.moves
        return snake
    
    def draw(self):
        for x, y in self.history:
            pygame.draw.rect(screen, (255, 255, 255), (x * SCALE, y * SCALE, SCALE, SCALE))
        pygame.draw.rect(screen, (0, 255, 0), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))

    def __str__(self):
        return f"Score: {self.score}, Fitness: {self.fitness}"
    
    def __repr__(self):
        return self.__str__()

# Create the food
class Food:
    def __init__(self):
        self.x = random.randint(2, WIDTH - 3)
        self.y = random.randint(2, HEIGHT - 3)

    def respawn(self):
        self.x = random.randint(2, WIDTH - 3)
        self.y = random.randint(2, HEIGHT - 3)

    def draw(self):
        pygame.draw.rect(screen, (255, 0, 0), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))
    
    def __str__(self):
        return f"Food: ({self.x}, {self.y})"
    
    def __repr__(self):
        return self.__str__()

# Draw the grid
def draw_grid():
    for x in range(0, WIDTH * SCALE, SCALE):
        pygame.draw.line(screen, (255, 255, 255), (x, 0), (x, HEIGHT * SCALE))
    for y in range(0, HEIGHT * SCALE, SCALE):
        pygame.draw.line(screen, (255, 255, 255), (0, y), (WIDTH * SCALE, y))


if __name__ == "__main__":
    # Initialize pygame
    screen = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))
    clock = pygame.time.Clock()
    # Create the population
    population = Population(100, 4, 8, 4)
    # Create the food
    food = Food()
    # Create the snakes
    snakes = [Snake(10, 10, population.get_random()) for _ in range(population.size)]

    # Main loop
    try:
        while True:
            # Check if all the snakes are dead
            if all([snake.dead for snake in snakes]):
                # Calculate the fitness of each snake
                for snake in snakes:
                    snake.calculate_fitness()
                # Natural selection
                population.natural_selection()
                # Create new snakes
                snakes = [Snake(10, 10, population.get_random()) for _ in range(population.size)]
                # Create new food
                food = Food()
                # Print the best snake
                print(population.get_best())

            # Loop through all the snakes
            for snake in snakes:
                # If the snake is not dead
                if not snake.dead:
                    # Think
                    snake.think(food)
                    # Update
                    if snake.update(food):
                        food = Food()
                    # Check if the snake is dead
                    if snake.is_dead():
                        snake.dead = True

            # Draw the background
            screen.fill((0, 0, 0))
            # Draw the food
            food.draw()
            # Loop through all the snakes
            for snake in snakes:
                # If the snake is not dead
                if not snake.dead:
                    # Draw the snake
                    snake.draw()
            # Update the screen
            pygame.display.update()
            # Set the frame rate
            clock.tick(60)
    except KeyboardInterrupt:
        pygame.quit()

    