import numpy as np
import random
import pygame
import os

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
    _count = 0
    def __init__(self, nn, x=0, y=0):
        super().__init__(x, y)
        self.nn = nn
        # Track the fitness score of the snake
        self.fitness = 0
        # Extending position data
        self.direction = 0
        # Record snakes previous movements to length food_eaten
        self.body = []
        self.food_eaten = 1
        self.moves = 0
        self.dead = False
        self.id = Snake._count
        Snake._count += 1

    def think(self, food):
        # Capture positional data in numpy array
        inputs = np.array([self.x, self.y, food.x, food.y, self.boundary_distance()])
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

    def boundary_distance(self):
        return min([(WIDTH - 1) - self.x, self.x, (HEIGHT - 1) - self.y, self.y])

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
        if self.x < 0 or self.y < 0 or self.x >= WIDTH or self.y >= HEIGHT:
            # Game over
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
        self.fitness = ((self.moves - max([WIDTH, HEIGHT])) * 25)**2 - (self.moves / self.food_eaten * 10)

    def is_dead(self):
        if (self.x, self.y) in self.body[:-1]:
            return True
        else:
            return False

    def clone(self):
        snake = Snake(self.nn.clone(), self.x, self.y)
        snake.fitness = self.fitness
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
        return f"Snake ID: {self.id} \nFitness: {self.fitness}\nMoves: {self.moves}\nFood Eaten: {self.food_eaten}\nMoves-Food Ratio: {self.moves / self.food_eaten}"
    
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
        rX = random.randint(2, WIDTH - 3)
        rY = random.randint(2, HEIGHT - 3)
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

class Population:
    def __init__(self, size: int, input_nodes, hidden_nodes, output_nodes, mutation_rate: float):
        self.size = size
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.mutation_rate = mutation_rate
        self.generation = 0
        
        self.population = []
        for i in range(self.size):
            nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
            snake = Snake(nn=nn)
            self.population.append(snake)

        self.best = self.get_fittest()

    def get_fitnesses(self):
        return np.array([snake.fitness for snake in self.population])
    
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
            future_generation.append(child)

        # Refresh current population
        self.population = future_generation
        # Increment generation count by 1
        self.generation += 1

    def select_parent(self) -> Snake:
        index = 0
        r = random.uniform(0, self.best.fitness)

        while r > 0:
            r -= self.population[index].fitness
            index += 1

        index -= 1
        return self.population[index]


if __name__ == "__main__":
    # Initialize pygame
    screen = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))
    clock = pygame.time.Clock()
    # Create the population
    population = Population(100, 5, 8, 4, 0.1)
    # Create the food
    food = Food()
    # Create the snakes
    snakes = [Snake.duplicate(population.get_random()) for _ in range(population.size)]

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
                snakes = [population.get_random() for _ in range(population.size)]
                # Create new food
                food = Food()

            # Loop through all the snakes
            for snake in snakes:
                # If the snake is not dead
                if not snake.dead:
                    # Think
                    snake.think(food)
                    # Update
                    snake.update(food)

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
            print(population.get_fittest())
            # Update the screen
            pygame.display.update()
            # Set the frame rate
            clock.tick(60)
    except KeyboardInterrupt:
        pygame.quit()
