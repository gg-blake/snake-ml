from python import Python
from snake import Snake
from collections import Set, Dict
from neural_network import NeuralNetwork
from algorithm.sort import sort

alias dtype = DType.float32
alias nn_dtype = NeuralNetwork[12, 16, 4]
alias game_width: Int = 20
alias game_width_offset: Int = game_width // 2
alias game_height: Int = 20
alias game_height_offset: Int = game_height // 2
alias starting_score: Int = 5

struct Population[snake_count: Int]:
    var habitat: DynamicVector[Snake]
    var food_array: DynamicVector[(Int, Int)]
    var active: Bool
    var generation: Int

    fn __init__(inout self) raises:
        self.habitat = DynamicVector[Snake]()
        self.food_array = DynamicVector[(Int, Int)]()
        self.active = True
        self.generation = 0
        self.generate_food()
        for id in range(snake_count):
            var snake_object = Snake(id)
            self.habitat.append(snake_object)

    fn generate_food(inout self) raises:
        var pyrandom = Python.import_module("random")
        var rand_x = pyrandom.randint(-game_width_offset, game_width_offset).to_float64().to_int()
        var rand_y = pyrandom.randint(-game_height_offset, game_height_offset).to_float64().to_int()
        self.food_array.append((rand_x, rand_y))

    fn update_habitat(inout self) raises -> Bool:
        print("Updating...")
        var survived = False

        for snake_reference in self.habitat:
            var snake = snake_reference[]
            if snake.score >= len(self.food_array):
                self.generate_food()
            var current_snake_fruit = self.food_array[snake.score - starting_score]
            var current_x_position_fruit = current_snake_fruit.get[0, Int]()
            var current_y_position_fruit = current_snake_fruit.get[1, Int]()
            snake.think(current_x_position_fruit, current_y_position_fruit)

            var distance: Float32 = Population[snake_count].euclidean_distance(snake.x_position, 
                snake.y_position, 
                current_x_position_fruit, 
                current_y_position_fruit)

            if distance < snake.min_dist:
                snake.min_dist = distance

            if snake.active:
                survived = True

            

        return survived

    @staticmethod
    fn fitness(snake: Snake) -> Float32:
        return ((400 / (snake.min_dist + 1)) + (snake.score**2 * 20))

    @staticmethod
    fn euclidean_distance(x1: Int, y1: Int, x2: Int, y2: Int) -> Float32:
        return ((x1 - x2)**2 + (y1 - y2)**2)**(0.5)


    fn generate_next_habitat(inout self, survival_rate: Float32) raises:
        # I don't know how to implement Mojo built-in sort so this is my work around :(
        var fitness_dict = Dict[Int, Int]()
        var fitness_array = DynamicVector[Int]()
        var index = 0
        for snake_reference in self.habitat:
            var current_snake: Snake = snake_reference[]
            var current_snake_fitness: Int = Population[snake_count].fitness(current_snake).to_int()

            fitness_dict[current_snake_fitness] = index
            fitness_array.append(current_snake_fitness)
            index += 1

        sort(fitness_array)
        print("Saved Fitness Scores")
        index = 0

        # Resurrect the fittest snakes, their objects are preserved in the habitat
        for index in range(snake_count):
            var habitat_index = fitness_dict[fitness_array[index]]
            self.habitat[habitat_index].active = True

        # For every two fit snakes, create two children with blended weights and biases and replace two unfit snake objects in the habitat
        for index in range(snake_count * survival_rate, snake_count, 2):
            var habitat_index = fitness_dict[fitness_array[index]]
            var fit_traits_a: nn_dtype = self.habitat[fitness_dict[fitness_array[index - (snake_count // 2)]]].neural_network
            var fit_traits_b: nn_dtype = self.habitat[fitness_dict[fitness_array[index - (snake_count // 2) + 1]]].neural_network
            var new_traits: nn_dtype = NeuralNetwork.blend_genetic_traits(fit_traits_a, fit_traits_b, self.generation * snake_count + index)
            var child: Snake = Snake(new_traits, self.generation * snake_count + index)
            self.habitat[habitat_index] = child
            self.habitat[habitat_index + 1] = child
            self.habitat[habitat_index + 1].id += 1
    