import numpy as np
import random
import pygame
import math
import time

WIDTH = 10
HEIGHT = 10
SCALE = 50

class GameObject:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Snake(GameObject):
    def __init__(self, x, y, nn):
        super().__init__(x, y)
        self.nn = nn
        # Track the fitness score of the snake
        self.fitness = 0
        # Extending position data
        self.direction = 0
        # Record snakes previous movements to length food_eaten
        self.body = []
        self.food_eaten = 0
        self.moves = 0
        self.alive = True

    def think(self, food):
        # Capture positional data in numpy array
        inputs = np.array([self.x, self.y, food.x, food.y])
        # Feed the food positional data and snake data into the neural net
        output = self.nn.feed(inputs)
        # Handle neural net output
        new_direction = np.argmax(output)
        # Ensure the snake doesn't turn in on itself
        # Probably a better implementation with binary operators but whateva
        if self.direction == 0 and new_direction == 2:
            return
        elif self.direction == 2 and new_direction == 0:
            return
        elif self.direction == 1 and new_direction == 3:
            return
        elif self.direction == 3 and new_direction == 1:
            return
        # Make the neural net direction the snake's direction
        self.direction = new_direction

    def update(self, food):
        self.moves += 1
        # Make a move based on snake's current direction
        if self.direction == 0:
            # Move right
            self.x += 1
        elif self.direction == 1:
            # Move down
            self.y += 1
        elif self.direction == 2:
            # Move left
            self.x -= 1
        elif self.direction == 3:
            # Move up
            self.y -= 1
        # Check for food collision
        if self == food:
            food.respawn()
            self.food_eaten += 1
        # Check for wall collision
        if self.x < 0 or self.y < 0 or self.x > WIDTH or self.y > HEIGHT:
            # Game over
            self.alive = False
        # Check for body collision
        if self.body.index((self.x, self.y)) != -1:
            # Game over
            self.alive = False
        # The body of the snake acts as a queue
        self.body.append((self.x, self.y))
        # The queue maintains a length of (food_eaten + 1) whenever a new move is made
        while len(self.body) > self.food_eaten + 1:
            # Shed the end of the tail
            self.body.pop(0)

    def calculate_fitness(self):
        self.fitness = self.food_eaten**2 * 100 - self.moves * 10


class Food(GameObject):
    def __init__(self, x, y):
        super().__init__(x, y)

    def respawn(self):
        rX = random.randint(0, WIDTH - 1)
        rY = random.randint(0, HEIGHT - 1)
        self.move(rX, rY)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Weights from input layer to hidden layer (ih)
        self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes)
        # Weights from hidden layer to output layer (ho)
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes)

        # Biases for hidden layer's nodes (h)
        self.bias_h = np.random.rand(self.hidden_nodes, 1)
        # Biases for output layer's nodes (o)
        self.bias_o = np.random.rand(self.output_nodes, 1)

    def feed(self, input_array):
        # Transpose the input array
        inputs = np.array(input_array, ndmin=2).T

        # Feed input layer to hidden layer
        hidden = np.dot(self.weights_ih, inputs)
        hidden = np.add(hidden, self.bias_h)
        hidden = self.sigmoid(hidden)

        # Feed hidden layer to output layer
        output = np.dot(self.weights_ho, hidden)
        output = np.add(output, self.bias_o)
        output = self.sigmoid(output)

        return output
    
    # Determines the activation of a node
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def mutate(self, mutation_rate):
        def mutate(val):
            if random.random() < mutation_rate:
                return val + random.gauss(0, 0.1)
            else:
                return val
            
        mutate = np.vectorize(mutate)

        # Set weights and biases to new mutated numpy array
        self.weights_ih = mutate(self.weights_ih)
        self.weights_ho = mutate(self.weights_ho)
        self.bias_h = mutate(self.bias_h)
        self.bias_o = mutate(self.bias_o)

    def clone(self):
        # Create new NeuralNetwork object with identical instance variables
        nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        # Copy the weights and biases
        nn.weights_ih = np.copy(self.weights_ih)
        nn.weights_ho = np.copy(self.weights_ho)
        nn.bias_h = np.copy(self.bias_h)
        nn.bias_o = np.copy(self.bias_o)
        return nn

    def __str__(self):
        return f"weights_ih: {self.weights_ih}, weights_ho: {self.weights_ho}, bias_h: {self.bias_h}, bias_o: {self.bias_o}"
    
    def __repr__(self):
        return self.__str__()

class Population:
    def __init__(self, size, input_nodes, hidden_nodes, output_nodes):
        self.size = size
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.generation = 0
        self.mutation_rate = 0.1
        self.best = None
        self.population = []
        for i in range(self.size):
            nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
            self.population.append(Snake(0, 0, nn))

    def get_fitnesses(self):
        return [snake.fitness for snake in self.population]
    
    def get_average_fitness(self):
        return sum(self.get_fitnesses()) / len(self.population)
    
    def get_fittest(self):
        min = 0
        min_index = -1
        for i, s in enumerate(self.get_fitnesses()):
            if s > min:
                min = s
                min_index = i

        return self.population[min_index]


    
    def natural_selection(self):
        new_population = []

        self.best = self.get_fittest()
        new_population.append(self.best.copy())
