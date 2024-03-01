from python import Python
from neural_network_sl import SingleLayerNeuralNetwork
from .snake_obj import Snake
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import json

struct SLPopulation:
    
    var size: Int
    var input_nodes: Int
    var hidden_nodes: Int
    var output_nodes: Int
    var mutation_rate: Float32
    var population: Snake[]
    var generation: Int
    var _nn_count: Int
    var np: object
    fn __init__(inout self, size: Int, input_nodes: Int, hidden_nodes: Int, output_nodes: Int, mutation_rate: Float32, population: Snake[]=None, generation: Int=0) raises:
        self.np = Python.import_module("numpy")
        # Instance Variables #
        self.size = size # Number of snakes in this population
        self.input_nodes = input_nodes # Number of nodes in input layer
        self.hidden_nodes = hidden_nodes # Number of nodes in hidden layer
        self.output_nodes = output_nodes # Number of nodes in output layer
        self.mutation_rate = mutation_rate # Mutation rate of this population
        self.generation = generation # Current generation of this population
        self._nn_count = Int(0) # Total number of SingleLayerNeuralNetwork instances
        # Set population list to None if we're not importing a JSON file
        if population is None:
            self.population = []
            # Create a new snake for each snake in the population
            for i in range(self.size):
                nn = SingleLayerNeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes, self._nn_count)
                self._nn_count += 1
                snake = Snake(nn=nn)
                self.population.append(snake)
        else:
            # Set population list to imported population
            self.population = population
        self.best = self.get_fittest() # Best snake in this population

    # String representation
    fn __str__(self):
        parents = " ".join([f"{parent[0]}x{parent[1]}" for parent in self.sort_parents_hist()])
        return f"Generation: {self.generation}\nAverage Fitness: {self.get_average_fitness()}\nBest Fitness: {self.best.fitness}\nAvg Distance: {self.get_average_distance()}\nAvg Moves: {self.get_average_moves()}\nParents:\n\t{parents}"
    
    # Printable representation
    fn __repr__(self):
        return self.__str__()
    
    # Dictionary representation
    fn __dict__(self):
        # This function enables models to be saved as JSON files
        return {
            "generation": self.generation,
            "population": [snake.nn.__dict__() for snake in self.population],
        }

    # Returns a list all the snake fitness scores in this population
    fn get_fitnesses(self):
        return [snake.fitness for snake in self.population]
    
    # Returns the average fitness of this population
    fn get_average_fitness(self) -> Float32:
        return sum(self.get_fitnesses()) / len(self.population)
    
    fn fitness_decay(self):
        for snake in self.population:
            snake.fitness -= int(self.get_average_fitness())
            if snake.fitness < 0:
                snake.fitness = 0

    fn get_average_distance(self) -> Float32:
        return sum([snake.calculate_avg_dist() for snake in self.population]) / len(self.population)
    
    fn get_average_moves(self) -> Float32:
        return sum([snake.moves for snake in self.population]) / len(self.population)
    
    # Returns the fittest snake in this population
    fn get_fittest(self) -> Snake:
        var max: Int = 0
        var max_index: Int = 0
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
        child_nn = SingleLayerNeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes, 0, parent_a.nn.id, parent_b.nn.id)
        child = Snake(nn=child_nn)

        # Child inherits average weights and bias of the parents
        child.nn.weights_ih = np.add(parent_a.nn.weights_ih, parent_b.nn.weights_ih) / Int(2)
        child.nn.weights_ho = np.add(parent_a.nn.weights_ho, parent_b.nn.weights_ho) / Int(2)
        child.nn.bias_h = np.add(parent_a.nn.bias_h, parent_b.nn.bias_h) / Int(2)
        child.nn.bias_o = np.add(parent_a.nn.bias_o, parent_b.nn.bias_o) / Int(2)

        return child
    
    def natural_selection(self) -> None:
        var future_generation: Snake[] = []

        # Fittest snake survives
        var best: Snake = self.get_fittest()
        
        
        # Add children to future generation
        for i in range(self.size):
            # Mix genetic composition of parents
            var parent_a: Snake = self.select_parent()
            var parent_b: Snake = self.select_parent()
            var child: Snake = self.breed(parent_a, parent_b)
            # Mutate adds potentially beneficial features
            child.nn.mutate(self.mutation_rate)
            future_generation.append(child)
                
        
        self.population = future_generation
        self.population[0] = best.clone() # No error here but might still be wrong
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

    fn get_parents(self) -> Int[]:
        var parents: Int[] = []
        for snake in self.population:
            parents.append(snake.nn.father)
            parents.append(snake.nn.mother)
        return parents
    
    fn get_parents_hist(self):
        var hist = Python.dict()
        for parent in self.get_parents():
            if parent in hist:
                hist[parent] += 1
            else:
                hist[parent] = 1
        return hist
    
    fn sort_parents_hist(self):
        var hist = self.get_parents_hist()
        # used to sort by id number but mojo doesn't support lambda functions yet
        return hist
    
    fn load_from_json(self, filename: String):
        if os.path.exists(f"./snake-ml/sl_data/{filename}"):
            with open(f"./snake-ml/sl_data/{filename}", "r") as f:
                data = json.load(f)

                self.generation = data["generation"]
                for i, v in enumerate(data["population"]):
                    self.population[i].nn.input_nodes = v['input_nodes']
                    self.population[i].nn.hidden_nodes = v['hidden_nodes']
                    self.population[i].nn.output_nodes = v['output_nodes']
                    self.population[i].nn.weights_ih = np.array(v['weights_ih'])
                    self.population[i].nn.weights_ho = np.array(v['weights_ho'])
                    self.population[i].nn.bias_h = np.array(v['bias_h'])
                    self.population[i].nn.bias_o = np.array(v['bias_o'])

                return self
            
        return None
    
    fn save_to_json(self, filename: String):
        # Save population.population to json
        with open(f"./snake-ml/sl_data/{filename}", "w") as f:
            json.dump(self.__dict__(), f)