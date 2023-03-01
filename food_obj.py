from .game_obj import GameObject
from .constants import *
from .snake import screen
import random, pygame

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
        if self.id == Food._count:
            pygame.draw.rect(screen, (255, 0, 0), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))
        else:
            pygame.draw.rect(screen, (80, 80, 80), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))