from python import Python
from memory.unsafe import Pointer
from collections.vector import InlinedFixedVector

struct Snake:
    var x: Int
    var y: Int

    fn __init__(inout self, x: Int, y: Int):
        self.x = x
        self.y = y

struct Population:
    var snake_count: Int
    var snakes: Snake
    fn __init__(inout self, snake_count: Int):
        self.snake_count = snake_count
        self.snakes = Snake(0, 0)