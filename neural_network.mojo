from python import Python
from matrix import Matrix, fast_dot_product, fast_sigmoid, fast_add, scale, average
from population import dtype

@value
struct NeuralNetwork[input_size: Int, hidden_size: Int, output_size: Int](Hashable):
    var input_to_hidden_weights: Matrix[hidden_size, input_size]
    var hidden_to_output_weights: Matrix[output_size, hidden_size]
    var hidden_node_biases: Matrix[hidden_size, 1]
    var output_node_biases: Matrix[output_size, 1]
    
    fn __init__(inout self):
        self.input_to_hidden_weights = Matrix[hidden_size, input_size].rand()
        self.hidden_to_output_weights = Matrix[output_size, hidden_size].rand()
        self.hidden_node_biases = Matrix[hidden_size, 1].rand()
        self.output_node_biases = Matrix[output_size, 1].rand()

    fn __init__(inout self, 
            ih: Matrix[hidden_size, input_size], 
            ho: Matrix[output_size, hidden_size], 
            h: Matrix[hidden_size, 1], 
            o: Matrix[output_size, 1]):
        self.input_to_hidden_weights = ih
        self.hidden_to_output_weights = ho
        self.hidden_node_biases = h
        self.output_node_biases = o

    fn __hash__(self) -> Int:
        return hash(self.input_to_hidden_weights)

    fn __copyinit__(inout self, existing: Self):
        self = NeuralNetwork(existing.input_to_hidden_weights, 
                             existing.hidden_to_output_weights, 
                             existing.hidden_node_biases, 
                             existing.output_node_biases)

    fn feed(self, input_array: Matrix[input_size, 1]) -> DTypePointer[dtype]:
        var hidden_array = Matrix[hidden_size, 1]()

        fast_dot_product(hidden_array, self.input_to_hidden_weights, input_array)
        fast_add(hidden_array, hidden_array, self.hidden_node_biases)
        fast_sigmoid(hidden_array, hidden_array)

        var output_array = Matrix[output_size, 1]()

        fast_dot_product(output_array, self.input_to_hidden_weights, hidden_array)
        fast_add(output_array, output_array, self.hidden_node_biases)
        fast_sigmoid(output_array, output_array)

        hidden_array.data.free()
        output_array.data.free()

        return output_array.data

    fn blend_genetic_traits(self: Self, other: Self) -> Self:
        var avg_matrix_ih = Matrix[hidden_size, input_size]()
        var avg_matrix_ho = Matrix[output_size, hidden_size]()
        var avg_matrix_h = Matrix[hidden_size, 1]()
        var avg_matrix_o = Matrix[output_size, 1]()
        average(avg_matrix_ih, self.input_to_hidden_weights, other.input_to_hidden_weights)
        average(avg_matrix_ho, self.hidden_to_output_weights, other.hidden_to_output_weights)
        average(avg_matrix_h, self.hidden_node_biases, other.hidden_node_biases)
        average(avg_matrix_o, self.output_node_biases, other.output_node_biases)
        # TODO: Mutate the weights and biases before return
        return NeuralNetwork(avg_matrix_ih, avg_matrix_ho, avg_matrix_h, avg_matrix_o)
        









        

    