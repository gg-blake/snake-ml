import numpy as np
import random
import pygame
import os
import json

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
    _count = 0
    def __init__(self, nn, x=int(WIDTH/2), y=int(HEIGHT/2)):
        super().__init__(x, y)
        self.nn = nn
        # Track the fitness score of the snake
        self.fitness = 1
        # Extending position data
        self.direction = 0
        # Record snakes previous movements to length food_eaten
        self.body = []
        self.food_eaten = 1
        self.moves = 1
        self.hungry_moves = 1
        self.dead = False
        self.death_dist = 0
        self.id = Snake._count
        Snake._count += 1

    @staticmethod
    def inBounds(x: int, y: int) -> bool:
        return x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT

    def think(self, food):
        # Capture positional data in numpy array
        inputs = np.array([self.x, self.y, food.x, food.y, Snake.inBounds(self.x - 1, self.y), Snake.inBounds(self.x + 1, self.y), Snake.inBounds(self.x, self.y - 1), Snake.inBounds(self.x, self.y + 1)])
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
    
    def food_distance(self, food):
        return ((self.x - food.x)**2 + (self.y - food.y)**2)**(1/2)

    def update(self, food):
        self.moves += 1
        self.hungry_moves += 1
        self.think(food)

        if self.dead:
            self.death_dist = self.food_distance(food)

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
            self.hungry_moves = 1
        # Check for wall collision
        if self.x < 0 or self.y < 0 or self.x >= WIDTH or self.y >= HEIGHT:
            # Game over
            self.dead = True
            return
        
        if self.hungry_moves > WIDTH * HEIGHT:
            self.dead = True
            return

        # Check for body collision
        self.dead = self.is_dead()

        # The body of the snake acts as a queue
        self.body.append((self.x, self.y))
        # The queue maintains a length of (food_eaten + 1) whenever a new move is made
        while len(self.body) > self.food_eaten + 1:
            # Shed the end of the tail
            self.body.pop(0)

        

    def calculate_fitness(self):
        self.fitness = (self.food_eaten * 100) + (self.moves * 50)

    def is_dead(self):
        if (self.x, self.y) in self.body[:-1]:
            return True
        else:
            return False

    def clone(self):
        snake = Snake(self.nn.clone(), self.x, self.y)
        snake.direction = self.direction
        snake.body = self.body.copy()
        snake.dead = self.dead
        snake.food_eaten = self.food_eaten
        snake.moves = self.moves
        return snake
    
    def draw(self):
        for x, y in self.body:
            pygame.draw.rect(screen, (255, 255, 255), (x * SCALE, y * SCALE, SCALE, SCALE))
        pygame.draw.rect(screen, (0, 255, 0), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))

    def __str__(self):
        return f"Snake ID: {self.id} \nFitness: {self.fitness}\nMoves: {self.moves}\nFood Eaten: {self.food_eaten}\nParents: {self.nn.parents}"
    
    def __repr__(self):
        return self.__str__()
    
    @classmethod
    def duplicate(cls, snake):
        return Snake(snake.nn, snake.x, snake.y)


class Food(GameObject):
    def __init__(self, x=0, y=0):
        super().__init__(x, y)
        self.respawn()

    def respawn(self):
        rX = random.randint(0, WIDTH - 1)
        rY = random.randint(0, HEIGHT - 1)
        self.move(rX, rY)

    def draw(self):
        pygame.draw.rect(screen, (255, 0, 0), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))
    
    def __str__(self):
        return f"Food: ({self.x}, {self.y})"
    
    def __repr__(self):
        return self.__str__()


class NeuralNetwork:
    _count = 0
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.id = NeuralNetwork._count
        NeuralNetwork._count += 1
        self.parents = [0, 0]

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
        nn.parents = [self.id, self.id]
        return nn

    def __str__(self):
        return f"weights_ih: {self.weights_ih}, weights_ho: {self.weights_ho}, bias_h: {self.bias_h}, bias_o: {self.bias_o}"
    
    def __repr__(self):
        return self.__str__()
    
    def __dict__(self):
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
        self.size = size
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.mutation_rate = mutation_rate
        self.generation = 0
        if population is None:
            self.population = []
        else:
            self.population = population
        for i in range(self.size):
            nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
            snake = Snake(nn=nn)
            self.population.append(snake)

        self.best = self.get_fittest()

    def __str__(self):
        return f"Generation: {self.generation}, Average Fitness: {self.get_average_fitness()}, Best Fitness: {self.best.fitness}"
    
    def __repr__(self):
        return self.__str__()
    
    def __dict__(self):
        return {
            "generation": self.generation,
            "population": [snake.nn.__dict__() for snake in self.population],
        }

    def get_fitnesses(self):
        return [snake.fitness for snake in self.population]
    
    def get_average_fitness(self) -> float:
        return sum(self.get_fitnesses()) / len(self.population)
    
    def get_fittest(self) -> Snake:
        max = 0
        max_index = 0
        for i, s in enumerate(self.get_fitnesses()):
            if s > max:
                max = s
                max_index = i

        return self.population[max_index]

    def get_random(self) -> Snake:
        return random.choice(self.population)

    def sexy_time(self, parent_a: Snake, parent_b: Snake) -> Snake:
        # Create new NeuralNetwork object
        child_nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        child = Snake(nn=child_nn)
        # Child inherits average weights and bias of the parents
        child.nn.weights_ih = np.add(parent_a.nn.weights_ih, parent_b.nn.weights_ih) / 2
        child.nn.weights_ho = np.add(parent_a.nn.weights_ho, parent_b.nn.weights_ho) / 2
        child.nn.bias_h = np.add(parent_a.nn.bias_h, parent_b.nn.bias_h) / 2
        child.nn.bias_o = np.add(parent_a.nn.bias_o, parent_b.nn.bias_o) / 2
        
        self.parents = [parent_a.nn.id, parent_b.nn.id]
        return child
    
    def natural_selection(self) -> None:
        future_generation = []

        # Fittest snake survives
        self.best = self.get_fittest()
        future_generation.append(self.best.clone())

        # Add children to future generation
        for i in range(self.size):
            # Mix genetic composition of parents
            parent_a: Snake = self.select_parent()
            parent_b: Snake = self.select_parent()
            child: Snake = self.sexy_time(parent_a, parent_b)
            # Mutate adds potentially beneficial features
            child.nn.mutate(self.mutation_rate)
            self.population[i] = child
        # Increment generation count by 1
        self.generation += 1

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
    # Load neural network from json
    



    # Initialize pygame
    screen = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))
    clock = pygame.time.Clock()
    # Create the population
    population = Population(100, 8, 64, 4, 0.5)
    # Create the food
    food = Food()
    # Load game state and preserve progress
    if os.path.exists("population.json"):
        with open("population.json", "r") as f:
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
   # Create the snakes
    snakes = [population.get_random() for _ in range(population.size)]
        

    try:
        while True:
            if population.generation % 50 == 0 and population.generation > 0:
                # Save population.population to json
                with open("population.json", "w") as f:
                    json.dump(population.__dict__(), f)

            # handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            # Check if all the snakes are dead
            if all([snake.dead for snake in snakes]):
                # Calculate the fitness of each snake
                for snake in snakes:
                    snake.calculate_fitness()
                # Natural selection
                population.natural_selection()
                # Create new snakes
                snakes = [population.get_random() for _ in range(population.size)]
                

            # Loop through all the snakes
            for snake in snakes:
                # If the snake is not dead
                if not snake.dead:
                    
                    # Update
                    snake.update(food)
                    # Check if snake found food
                    if snake == food:
                        # Create new food
                        food.respawn()

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

            # Print Population Stats
            os.system('cls')
            print(population)
            # Update the screen
            pygame.display.update()
            # Set the frame rate
            clock.tick(60)
    except KeyboardInterrupt:
        pygame.quit()
