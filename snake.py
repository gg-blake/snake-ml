import numpy as np
import random
import pygame
import os
import json
import sys
import time

WIDTH = 20
HEIGHT = 20
SCALE = 25
MAX_DISTANCE = (WIDTH**2 + HEIGHT**2)**(1/2)



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
    _count = 0 # Total number of Snake instances
    _active_food = [] # Current food active in the generation
    _starting_food = 6 # THIS IS FOR DEBUGGING PURPOSES ONLY

    def __init__(self, nn, x=int(WIDTH/2), y=int(HEIGHT/2)):
        # Inherited Instance Variables from GameObject #
        super().__init__(x, y)

        # Unique Instance Variables #
        self.nn = nn # Stores the snakes brain as a NeuralNetwork object
        self.fitness = 1 # Tracks the fitness score of the snake
        self.direction = 0 # Provide extra positional data
        self.body = [] # History of snake's previous positions up to snake's length (food_eaten), acts as a queue
        self.food_eaten = Snake._starting_food # Tracks the number of food eaten upon birth
        self.moves = 1 # Tracks the number of steps made since birth
        self.hungry_moves = 1 # Tracks number of steps made since last meal
        self.dead = False # Tracks if snake is dead or alive
        self.min_distance = (WIDTH**2 + HEIGHT**2)**(1 / 2) # Tracks the closest euclidean distance between the snake and the target food
        self.death_dist = 0 # Tracks the euclidean distance between the snake and target food upon death
        self.id = Snake._count # Unique Snake object identifier

        # Instance Callbacks #
        self.update()
        Snake._count += 1 # Increment the number of total snakes by 1

    # String representation
    def __str__(self):
        return f"Snake ID: {self.id} \nFitness: {self.fitness}\nMoves: {self.moves}\nFood Eaten: {self.food_eaten}\nParents: {self.nn.parents}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()

    # Returns whether or not the given 2D coordinates lie within the game space
    @staticmethod
    def inBounds(x: int, y: int) -> bool:
        return x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT
    
    # Unfair coinflip; equivalent to bernoulli
    @staticmethod
    def entropy(probabiltiy_threshold: float, probability_max:int=100):
        probability = random.randint(0, probability_max) / probability_max
        return probability > probabiltiy_threshold

    # Compute the snake's next move
    def think(self, food):
        # Capture positional data in numpy array
        inputs = np.array([
            self.direction, 
            self.x, 
            self.y, 
            food.x, 
            food.y, 
            Snake.inBounds(self.x - 1, self.y), 
            Snake.inBounds(self.x + 1, self.y), 
            Snake.inBounds(self.x, self.y - 1), 
            Snake.inBounds(self.x, self.y + 1)
        ])
        # Feed the food positional data and snake data into the neural net
        output = self.nn.feed(inputs)
        # Handle neural net output
        new_direction = np.argmax(output)
        # Ensure the snake doesn't turn in on itself
        # Probably a better implementation with binary operators but whateva
        while self.direction == 0 and new_direction == 2 or self.direction == 2 and new_direction == 0 or self.direction == 1 and new_direction == 3 or self.direction == 3 and new_direction == 1:
            # Capture positional data in numpy array
            inputs = np.array([
                self.direction, 
                self.x, 
                self.y, 
                food.x, 
                food.y, 
                Snake.inBounds(self.x - 1, self.y), 
                Snake.inBounds(self.x + 1, self.y), 
                Snake.inBounds(self.x, self.y - 1), 
                Snake.inBounds(self.x, self.y + 1)
            ])
            # Feed the food positional data and snake data into the neural net
            output = self.nn.feed(inputs)
            # Handle neural net output
            new_direction = np.argmax(output)
            # Randomness factor
            if Snake.entropy(0.95):
                new_direction = random.randint(0, 3)
        # Make the neural net direction the snake's direction
        self.direction = new_direction
    
    # Compute the euclidean distance between two GameObjects
    def food_distance(self, food):
        return ((self.x - food.x)**2 + (self.y - food.y)**2)**(1/2)
    
    # Returns the snake's score
    def calculate_score(self):
        '''
        DEBUG NOTE:
        Because we don't want more than 1 food to be present when the starting food score is higher than 1, 
        we must offset the food eaten with our initial food score constant.
        '''
        return self.food_eaten - Snake._starting_food

    # Perform game logic from one step
    def update(self):
        # Verify that there is another available active Food object at this snake's food level
        # Keep appending another Food object to the game's list of active food sources until there is another available active Food object at this snake's food level
        while len(Snake._active_food) <= self.calculate_score():
            Snake._active_food.append(Food.spawn())

        # Update this snake's target food
        self.target_food = Snake._active_food[self.calculate_score()]

        # Increment this snake's number of moves and moves w/o food by 1
        self.moves += 1
        self.hungry_moves += 1

        # Make a decision
        self.think(self.target_food)

        # Make a move based on this snake's current direction
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
        if self == self.target_food:
            # Increment this snake's total number of moves by 1
            self.food_eaten += 1
            # Reset this snake's number of hungry moves to 1
            self.hungry_moves = 1
        
        # Update the closest this snake has been to it's target food
        dist = self.food_distance(self.target_food)
        if dist < self.min_distance:
            self.min_distance = dist

        # Update whether or not this snake has died
        if self.is_dead():
            self.dead = True
            # Record the euclidean distance of snake on death
            self.death_dist = self.food_distance(self.target_food)
            return
        
        # Add this snake's new position to its body
        self.body.append((self.x, self.y))
        # Reduce this snake body's 
        while len(self.body) > self.food_eaten + 1:
            # Shed the end of the tail
            self.body.pop(0)

        
    # Returns and sets the fitness score of this snake
    def calculate_fitness(self) -> int:
        self.fitness = (self.food_eaten**2 * 100) + (self.moves * 50) - (int((self.min_distance + self.death_dist) / 2) * 4)
        return self.fitness

    # Returns whether or not the snake is dead
    def is_dead(self) -> bool:
        touching_body = (self.x, self.y) in self.body[:-1] # Die if touching self
        starve = self.hungry_moves > WIDTH * HEIGHT # Die if too many moves w/o food
        out_of_bounds = not Snake.inBounds(self.x, self.y) # Die if snake goes out of game bounds
        return touching_body or starve or out_of_bounds

    # Returns an identical Snake object (minus fitness score)
    def clone(self):
        # Construct Snake object with cloned NeuralNetwork object, and this snake's game coordinates
        snake = Snake(self.nn.clone(), self.x, self.y)
        # Mirror the direction
        snake.direction = self.direction
        # Mirror the body
        snake.body = self.body.copy()
        # Mirror the dead status
        snake.dead = self.dead
        # Mirror 
        snake.food_eaten = self.food_eaten
        snake.moves = self.moves
        return snake
    
    # Draws visual representation of this Snake object to the running pygame window
    def draw(self):
        # Snakes that have eaten the most food off their generation are the brightest
        main_wieght = int((self.calculate_score() + 1) / len(Snake._active_food) * 255)
        
        # Draw the body
        count = 0
        for x, y in self.body[1:]:
            count += 1
            # Points in the body get darker the closer they are to the end
            tail_weight = int(count / len(self.body) * 64 + int((self.calculate_score() + 1) / len(Snake._active_food) * 128))
            # Draw rect to screen
            pygame.draw.rect(screen, (tail_weight, tail_weight, tail_weight), (x * SCALE, y * SCALE, SCALE, SCALE))
            
        # Draw the head of the snake
        pygame.draw.rect(screen, (0, main_wieght, 0), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))


class Food(GameObject):
    _count = 0 # Total number of Food instances

    def __init__(self, x, y):
        # Inherited Instance Variables from GameObject #
        super().__init__(x, y)

        # Instance Callbacks #
        Food._count += 1 # Increment total number of Food objects by 1
        # Unique Instance Variable #
        self.id = Food._count # Unique Food object identifier

        
        
    # String representation
    def __str__(self):
        return f"Food: ({self.x}, {self.y})"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()

    # Returns an Food object with randomized x and y values
    @classmethod
    def spawn(cls):
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)
        return cls(x, y)
        
    # Draws visual representation of this Food object to the running pygame window
    def draw(self):
        # Draw rect to screen
        pygame.draw.rect(screen, (int(self.id/Food._count*255), 0, 0), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))
    
    


class NeuralNetwork:
    _count = 0 # Total number of NeuralNetwork instances

    def __init__(self, input_nodes, hidden_nodes, output_nodes, mother=0, father=0):
        # Instance Variables #
        self.input_nodes = input_nodes # Number of nodes in input layer
        self.hidden_nodes = hidden_nodes # Number of nodes in hidden layer
        self.output_nodes = output_nodes # Number of nodes in output layer
        self.id = NeuralNetwork._count # Unique NeuralNetwork object identifier
        self.mother = mother # NeuralNetwork id of mother
        self.father = father # NeuralNetwork id of father
        self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes) # Weights from input layer to hidden layer
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes) # Weights from hidden layer to output layer
        self.bias_h = np.random.rand(self.hidden_nodes, 1) # Biases of hidden layer
        self.bias_o = np.random.rand(self.output_nodes, 1) # Biases of output layer

        # Instance Callback #
        NeuralNetwork._count += 1 # Increment total number of NeuralNetwork objects by 1

    # Perform a feedforward operation on this NeuralNetwork object and return the output
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

    # Potentially mutate the weights and biases of this NeuralNetwork object
    def mutate(self, mutation_rate):
        # Vectorized function to mutate a value
        def mutate(val):
            # Mutate value if random number is less than mutation rate
            if random.random() < mutation_rate:
                return val + random.gauss(0, 0.1)
            else:
                return val
            
        # Vectorize the mutate function
        mutate = np.vectorize(mutate)

        # Set weights and biases to new mutated numpy array
        self.weights_ih = mutate(self.weights_ih)
        self.weights_ho = mutate(self.weights_ho)
        self.bias_h = mutate(self.bias_h)
        self.bias_o = mutate(self.bias_o)

    # Returns a copy of this NeuralNetwork object
    def clone(self):
        # Create new NeuralNetwork object with identical instance variables
        nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        # Copy the weights and biases
        nn.weights_ih = np.copy(self.weights_ih)
        nn.weights_ho = np.copy(self.weights_ho)
        nn.bias_h = np.copy(self.bias_h)
        nn.bias_o = np.copy(self.bias_o)
        nn.mother, nn.father = self.id, self.id
        return nn

    # String representation
    def __str__(self):
        return f"weights_ih: {self.weights_ih}, weights_ho: {self.weights_ho}, bias_h: {self.bias_h}, bias_o: {self.bias_o}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()
    
    # Dictionary representation
    def __dict__(self):
        # This function enables models to be saved as JSON files
        return {
            "input_nodes": self.input_nodes,
            "hidden_nodes": self.hidden_nodes,
            "output_nodes": self.output_nodes,
            "weights_ih": self.weights_ih.tolist(),
            "weights_ho": self.weights_ho.tolist(),
            "bias_h": self.bias_h.tolist(),
            "bias_o": self.bias_o.tolist()
        }

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
        return f"Generation: {self.generation}, Average Fitness: {self.get_average_fitness()}, Best Fitness: {self.best.fitness}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()
    
    # Dictionary representation
    def __dict__(self):
        # This function enables models to be saved as JSON files
        return {
            "generation": self.generation,
            "population": [dict(snake.nn) for snake in self.population],
        }

    # Returns a list all the snake fitness scores in this population
    def get_fitnesses(self):
        return [snake.fitness for snake in self.population]
    
    # Returns the average fitness of this population
    def get_average_fitness(self) -> float:
        return sum(self.get_fitnesses()) / len(self.population)
    
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
        # Create a new population
        future_generation = []

        # Fittest snake survives
        self.best = self.get_fittest()
        future_generation.append(self.best.clone())

        # Add children to future generation
        for i in range(self.size):
            # Mix genetic composition of parents
            parent_a: Snake = self.select_parent()
            parent_b: Snake = self.select_parent()
            child: Snake = self.breed(parent_a, parent_b)
            # Mutate adds potentially beneficial features
            child.nn.mutate(self.mutation_rate)
            future_generation.append(child)
        
        # Increment generation count by 1
        self.generation += 1

    # Selects a parent weighted by fitness
    def select_parent(self) -> Snake:
        probabilities = map(lambda snake: random.randint(1, snake.fitness + 1), self.population)
        max = 1
        max_i = 0
        for i, m in enumerate(probabilities):
            if m > max:
                max = m
                max_i = i

        return self.population[max_i]


    
    
    

if __name__ == "__main__":
    # Set record mode
    write_interval = 0
    write_location = ""
    if sys.argv[1] == "record":
        write_mode = True
        write_interval = int(sys.argv[2])
        write_location = sys.argv[3]
    else:
        write_mode = False
    
    os.system("cls")
    print(f"Write mode: {write_mode}\nWrite interval: every {write_interval} generations\nReading/Write at: ./{write_location}")
    time.sleep(2)

    # Initialize pygame
    screen = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))
    clock = pygame.time.Clock()
    # Create the population
    population = Population(100, 9, 20, 4, 0.2)
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

    time.sleep(2)
   # Create the snakes
    snakes = [population.get_random() for _ in range(population.size)]
    while True:
        if population.generation % write_interval == 0 and population.generation > 0 and write_mode:
            # Save population.population to json
            with open(write_location, "w") as f:
                json.dump(population.__dict__(), f)

        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

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
