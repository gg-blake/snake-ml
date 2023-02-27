from .neural_network import NeuralNetwork
from .snake_obj import Snake
import random
import numpy as np
import matplotlib.pyplot as plt

class Population:
    def __init__(self, size: int, input_nodes, hidden_nodes, output_nodes, mutation_rate: float, population=None):
        # Instance Variables #
        self.size = size # Number of snakes in this population
        self.input_nodes = input_nodes # Number of nodes in input layer
        self.hidden_nodes = hidden_nodes # Number of nodes in hidden layer
        self.output_nodes = output_nodes # Number of nodes in output layer
        self.mutation_rate = mutation_rate # Mutation rate of this population
        self.generation = 0 # Current generation of this population
        # Set population list to None if we're not importing a JSON file
        if population is None:
            self.population = []
            # Create a new snake for each snake in the population
            for i in range(self.size):
                nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
                snake = Snake(nn=nn)
                self.population.append(snake)
        else:
            # Set population list to imported population
            self.population = population
        self.best = self.get_fittest() # Best snake in this population

    # String representation
    def __str__(self):
        parents = " ".join([f"{parent[0]}x{parent[1]}" for parent in self.sort_parents_hist()])
        return f"Generation: {self.generation}\nAverage Fitness: {self.get_average_fitness()}\nBest Fitness: {self.best.fitness}\nAvg Distance: {self.get_average_distance()}\nAvg Moves: {self.get_average_moves()}\nParents:\n\t{parents}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()
    
    # Dictionary representation
    def __dict__(self):
        # This function enables models to be saved as JSON files
        return {
            "generation": self.generation,
            "population": [snake.nn.__dict__() for snake in self.population],
        }

    # Returns a list all the snake fitness scores in this population
    def get_fitnesses(self):
        return [snake.fitness for snake in self.population]
    
    # Returns the average fitness of this population
    def get_average_fitness(self) -> float:
        return sum(self.get_fitnesses()) / len(self.population)
    
    def fitness_decay(self):
        for snake in self.population:
            snake.fitness -= int(self.get_average_fitness())
            if snake.fitness < 0:
                snake.fitness = 0

    def get_average_distance(self) -> float:
        return sum([snake.calculate_avg_dist() for snake in self.population]) / len(self.population)
    
    def get_average_moves(self) -> float:
        return sum([snake.moves for snake in self.population]) / len(self.population)
    
    # Returns the fittest snake in this population
    def get_fittest(self) -> Snake:
        max = 0
        max_index = 0
        for i, s in enumerate(self.get_fitnesses()):
            if s > max:
                max = s
                max_index = i

        return self.population[max_index]

    # Returns a random snake in this population
    def get_random(self) -> Snake:
        return random.choice(self.population)

    # Mix the genes of two snakes to create a new snake
    def breed(self, parent_a: Snake, parent_b: Snake) -> Snake:
        # Create new NeuralNetwork object
        child_nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        child = Snake(nn=child_nn)

        # Child inherits average weights and bias of the parents
        child.nn.weights_ih = np.add(parent_a.nn.weights_ih, parent_b.nn.weights_ih) / 2
        child.nn.weights_ho = np.add(parent_a.nn.weights_ho, parent_b.nn.weights_ho) / 2
        child.nn.bias_h = np.add(parent_a.nn.bias_h, parent_b.nn.bias_h) / 2
        child.nn.bias_o = np.add(parent_a.nn.bias_o, parent_b.nn.bias_o) / 2

        # Child custody
        child.nn.mother = parent_a.nn.id
        child.nn.father = parent_b.nn.id
        return child
    
    def natural_selection(self) -> None:
        future_generation = []

        # Fittest snake survives
        self.best = self.get_fittest()
        
        
        # Add children to future generation
        for i in range(self.size):
            # Mix genetic composition of parents
            parent_a: Snake = self.select_parent()
            parent_b: Snake = self.select_parent()
            child: Snake = self.breed(parent_a, parent_b)
            # Mutate adds potentially beneficial features
            child.nn.mutate(self.mutation_rate)
            future_generation.append(child)
                
        
        self.population = future_generation
        self.population[0] = self.best.clone()
        # Increment generation count by 1
        self.generation += 1

    # Selects a parent weighted by fitness
    def select_parent(self) -> Snake:
        deviations = [snake.fitness - int(self.get_average_fitness()) for snake in self.population]
        deviations = [deviation if deviation > 0 else 0 for deviation in deviations]
        probabilities = map(lambda d: random.randint(0, int(2**d)), deviations)
        max = 1
        max_i = 0
        for i, m in enumerate(probabilities):
            if m > max:
                max = m
                max_i = i

        return self.population[max_i]

    def get_parents(self):
        parents = []
        for snake in self.population:
            parents.append(snake.nn.father)
            parents.append(snake.nn.mother)
        return parents
    
    def get_parents_hist(self):
        hist = {}
        for parent in self.get_parents():
            if parent in hist:
                hist[parent] += 1
            else:
                hist[parent] = 1
        return hist
    
    def sort_parents_hist(self):
        hist = self.get_parents_hist()
        return sorted(hist.items(), key=lambda x: x[1], reverse=True)