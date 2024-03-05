from algorithm.sort import sort
from matrix import Matrix
from population import dtype, nn_dtype, game_width, game_width_offset, game_height, game_height_offset

@value
struct Snake(CollectionElement, Hashable):
    var x_position: Int
    var y_position: Int
    var score: Int
    var history: DynamicVector[(Int, Int)]
    var active: Bool
    var min_dist: Float32
    var neural_network: nn_dtype

    fn __init__(inout self):
        self.x_position = 0
        self.y_position = 0
        self.score = 0
        self.history = DynamicVector[(Int, Int)]()
        self.history.resize(1, (self.x_position, self.y_position))
        self.active = True
        self.min_dist = game_width * game_height
        self.neural_network = nn_dtype()

    fn __init__(inout self, nn_data: nn_dtype):
        self.__init__()
        self.neural_network = nn_data

    fn __copyinit__(inout self, existing: Self):
        self = Snake(existing.neural_network)

    fn __moveinit__(inout self, owned existing: Self):
        self = Snake(existing.neural_network)

    fn __hash__(self) -> Int:
        return hash(self.neural_network)

    fn __eq__(self, other: Snake) -> Bool:
        return hash(self) == hash(other)

    fn inBounds(self) -> Int:
        if (self.x_position >= -game_width_offset and self.x_position <= game_width_offset and self.y_position >= -game_height_offset and self.y_position <= game_height_offset):
            return 1
        
        return 0

    @staticmethod
    fn inBounds(x: Int, y: Int) -> Int:
        if (x >= -game_width_offset and x <= game_width_offset and y >= -game_height_offset and y <= game_height_offset):
            return 1
        
        return 0

    fn inBody(self) -> Int:
        var count = 0
        for i in self.history:
            var current_pos = i.get_unsafe_pointer()[0]
            var current_pos_x = current_pos.get[0, Int]()
            var current_pos_y = current_pos.get[1, Int]()
            if current_pos_x == self.x_position and current_pos_y == self.y_position:
                count += 1

        if count > 1:
            return 1
        
        return 0

    @staticmethod
    fn inBody(history: DynamicVector[(Int, Int)], x: Int, y: Int) -> Int:
        var count = 0
        for i in history:
            var current_pos = i.get_unsafe_pointer()[0]
            var current_pos_x = current_pos.get[0, Int]()
            var current_pos_y = current_pos.get[1, Int]()
            if current_pos_x == x and current_pos_y == y:
                count += 1

        if count > 1:
            return 1
        
        return 0

    fn move(inout self, x: Int, y: Int):
        self.x_position = x
        self.y_position = y

    fn update(inout self, x_position_fruit: Int, y_position_fruit: Int):
        if self.inBody() or not self.inBounds():
            self.active = False
        elif self.x_position == x_position_fruit and self.y_position == y_position_fruit:
            self.score += 1
            self.min_dist = game_width * game_height

    fn think(inout self, x_position_fruit: Int, y_position_fruit: Int):
        if not self.active:
            return

        var input_array = VariadicList(self.x_position, 
                           self.y_position, 
                           x_position_fruit, 
                           y_position_fruit, 
                           Snake.inBounds(self.x_position - 1, self.y_position), 
                           Snake.inBounds(self.x_position + 1, self.y_position), 
                           Snake.inBounds(self.x_position, self.y_position - 1), 
                           Snake.inBounds(self.x_position, self.y_position + 1),
                           Snake.inBody(self.history, self.x_position - 1, self.y_position), 
                           Snake.inBody(self.history, self.x_position + 1, self.y_position), 
                           Snake.inBody(self.history, self.x_position, self.y_position - 1), 
                           Snake.inBody(self.history, self.x_position, self.y_position + 1))
        
        var input_matrix = Matrix[12, 1].encode(input_array) 
        var output_matrix: Matrix[4, 1] = self.neural_network.feed(input_matrix)
        var output_array = output_matrix.decode()
        sort(output_array)
        var previous_position = self.history[-2]

        for ptr in output_array:
            var choice = ptr.get_unsafe_pointer()[0]
            # Up
            if choice == 0 and previous_position.get[1, Int]() != self.y_position + 1:
                self.move(self.x_position, self.y_position + 1)
            # Left
            elif choice == 1 and previous_position.get[0, Int]() != self.y_position - 1:
                self.move(self.x_position - 1, self.y_position)
            # Down
            elif choice == 2 and previous_position.get[1, Int]() != self.y_position - 1:
                self.move(self.x_position, self.y_position - 1)
            # Right
            elif choice == 3 and previous_position.get[0, Int]() != self.y_position + 1:
                self.move(self.x_position + 1, self.y_position)

        self.update(x_position_fruit, y_position_fruit)

    

        



    

    


    