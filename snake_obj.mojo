from .game_obj import GameObject
from .food_obj import Food
from .constants import *
from .snake import screen
import random, pygame
import numpy as np
from neural_network_sl import SingleLayerNeuralNetwork


struct Snake:
    _count = 0 # Total number of Snake instances
    _active_food = [] # Current food active in the generation
    _starting_food = 6 # THIS IS FOR DEBUGGING PURPOSES ONLY
    var x: Int
    var y: Int
    var nn: SingleLayerNeuralNetwork
    var fitness: Int
    var direction: Int
    var body: Tuple[]
    var food_eaten: Int
    

    fn __init__(inout self, nn: SingleLayerNeuralNetwork, x: Int=int(Int(40)/2), y: Int=int(Int(40)/2)):
        # Inherited Instance Variables from GameObject #
        self.x = x
        self.y = y

        # Unique Instance Variables #
        self.nn = nn # Stores the snakes brain as a NeuralNetwork object
        self.fitness = 1 # Tracks the fitness score of the snake
        self.direction = 0 # Provide extra positional data
        self.body = [] # History of snake's previous positions up to snake's length (food_eaten), acts as a queue
        self.food_eaten = Snake._starting_food # Tracks the number of food eaten upon birth
        self.moves = 1 # Tracks the number of steps made since birth
        self.hungry_moves = 1 # Tracks number of steps made since last meal
        self.dead = False # Tracks if snake is dead or alive
        self.death_dist = 0 # Tracks the euclidean distance between the snake and target food upon death
        self.distances = [500] # Tracks the euclidean distance between the snake and target food at each step
        self.history = [] # Tracks the snake's position at each step
        self.hungry_history = []

        # Instance Callbacks #
        self.update()
        Snake._count += 1 # Increment the number of total snakes by 1

    # Inherited from GameObject (Mojo doesn't support struct inheritance)
    def move(self, x, y):
        self.x = x
        self.y = y

    # Inherited from GameObject (Mojo doesn't support struct inheritance)
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # String representation
    def __str__(self):
        return f"Snake ID: {self.nn.id} \nFitness: {self.fitness}\nMoves: {self.moves}\nFood Eaten: {self.food_eaten}\nParents: {self.nn.parents}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()

    # Returns whether or not the given 2D coordinates lie within the game space
    @staticmethod
    def inBounds(x: int, y: int) -> bool:
        return x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT

    # Compute the snake's next move
    def think(self, food):
        # Capture positional data in numpy array
        inputs = np.array([
            self.x, 
            self.y, 
            food.x, 
            food.y, 
            Snake.inBounds(self.x - 1, self.y), 
            Snake.inBounds(self.x + 1, self.y), 
            Snake.inBounds(self.x, self.y - 1), 
            Snake.inBounds(self.x, self.y + 1),
            (self.x + 1, self.y) in self.body,
            (self.x - 1, self.y) in self.body,
            (self.x, self.y + 1) in self.body,
            (self.x, self.y - 1) in self.body
        ])

        # Feed the food positional data and snake data into the neural net
        output = self.nn.feed(inputs)
        # Handle neural net output
        choices = np.argsort(output, axis=0)
        options = [1, 2, 3, 4]
        for i in range(len(choices)):
            '''touches = [GameObject(self.x + i, self.y + j) == self.target_food for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
            for j, is_touching in enumerate(touches):
                if is_touching:
                    options = []'''
            options[i] = choices[i]

        touches_food = [GameObject(self.x + i, self.y + j) == self.target_food for i, j in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        for i, is_touching in enumerate(touches_food):
            if is_touching:
                self.direction = i
                return

        
        for index, option in enumerate(options):
            if abs(self.direction - option) == 2:
                continue

            # Chance for random path change diminishes as you go down the list of options
            if random.random() < 0.05 * (1 / (index + 1)):
                try:
                    # Don't randomize path if next option hits body
                    if not [(self.x + i, self.y + j) in self.body for i, j in [(1, 0), (0, 1), (-1, 0), (0, -1)]][index + 1]:
                        continue
                except IndexError:
                    continue

            self.direction = option
            break

    
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
    
    # Returns and sets the fitness score of this snake
    def calculate_fitness(self) -> int:
        min_dist = min(self.distances)
        self.fitness = int(400 / (min_dist + 1)) + int(self.food_eaten**2 * 20)
        return self.fitness
    
    def calculate_avg_dist(self) -> float:
        return sum(self.distances) / len(self.distances)
    
    def eat(self):
        # Increment this snake's total number of moves by 1
        self.food_eaten += 1
        # Record number of hungry moves
        self.hungry_history.append(self.hungry_moves)
        # Reset this snake's number of hungry moves to 0
        self.hungry_moves = 0
        # Min distance becomes 0
        self.distances.append(0)
        # Reset history
        self.history = []

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
            self.eat()
            
        # Update the closest this snake has been to it's target food
        self.distances.append(self.food_distance(self.target_food))
        self.history.append((self.x, self.y))

        # Update whether or not this snake has died
        self.dead = self.is_dead()
        
        # Add this snake's new position to its body
        self.body.append((self.x, self.y))
        # Reduce this snake body's 
        while len(self.body) > self.food_eaten + 1:
            # Shed the end of the tail
            self.body.pop(0)

        
    

    # Returns whether or not the snake is dead
    def is_dead(self) -> bool:
        touching_body = (self.x, self.y) in self.body[:-1] # Die if touching self
        starve = self.hungry_moves > 10000000 # Die if too many moves w/o food
        out_of_bounds = not Snake.inBounds(self.x, self.y) # Die if snake goes out of game bounds
        loop_detected = self.loop_detection(3) # Die if snake loops back on itself
        return touching_body or starve or out_of_bounds or loop_detected

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
        for x, y in self.body:
            count += 1
            try:
                # Points in the body get darker the closer they are to the end
                tail_weight = int(count / len(self.body) * 32 + int((self.calculate_score() + 1) / len(Snake._active_food) * 128))
                # Draw rect to screen
                if self.calculate_score() + 1 >= len(Snake._active_food):
                    pygame.draw.rect(screen, (self.nn.id.r, self.nn.id.g, self.nn.id.b), (x * SCALE, y * SCALE, SCALE, SCALE))
                else:
                    pygame.draw.rect(screen, (20, 20, 20), (x * SCALE, y * SCALE, SCALE, SCALE))
            except ValueError:
                pass
            
        # Draw the head of the snake
        try:
            if self.calculate_score() + 1 >= len(Snake._active_food):
                pygame.draw.rect(screen, (self.nn.id.r, 255, self.nn.id.b), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))
            else:
                pygame.draw.rect(screen, (60, 60, 60), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))
        except ValueError:
            pass

    def loop_detection(self, threshold) -> bool:
        return self.history.count((self.x, self.y)) > threshold

