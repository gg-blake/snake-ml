from python import Python
from matrix import Matrix, fast_dot_product, fast_sigmoid, fast_add

alias dtype = DType.float32

struct NeuralNetwork[input_size: Int, hidden_size: Int, output_size: Int]:
    var input_to_hidden_weights: Matrix[dtype, hidden_size, input_size]
    var hidden_to_output_weights: Matrix[dtype, output_size, hidden_size]
    var hidden_node_biases: Matrix[dtype, hidden_size, 1]
    var output_node_biases: Matrix[dtype, output_size, 1]
    
    fn __init__(inout self):
        self.input_to_hidden_weights = Matrix[dtype, hidden_size, input_size].rand()
        self.hidden_to_output_weights = Matrix[dtype, output_size, hidden_size].rand()
        self.hidden_node_biases = Matrix[dtype, hidden_size, 1].rand()
        self.output_node_biases = Matrix[dtype, output_size, 1].rand()

    fn feed(self, input_array: Matrix[dtype, input_size, 1]) -> DTypePointer[dtype]:
        var hidden_array = Matrix[dtype, hidden_size, 1]()

        fast_dot_product[dtype](hidden_array, self.input_to_hidden_weights, input_array)
        fast_add[dtype](hidden_array, hidden_array, self.hidden_node_biases)
        fast_sigmoid[dtype](hidden_array, hidden_array)

        var output_array = Matrix[dtype, output_size, 1]()

        fast_dot_product[dtype](output_array, self.input_to_hidden_weights, hidden_array)
        fast_add[dtype](output_array, output_array, self.hidden_node_biases)
        fast_sigmoid[dtype](output_array, output_array)

        hidden_array.data.free()
        output_array.data.free()

        return output_array.data


fn main():
    var nn = NeuralNetwork[8, 12, 4]()

    var input = Matrix[dtype, 8, 1].rand()
    var output = Matrix[dtype, 4, 1]()

    output.data = nn.feed(input)

    var C = Matrix[DType.float32, 2, 2]()
    var A = Matrix[DType.float32, 2, 3].rand()
    var B = Matrix[DType.float32, 3, 2].rand()

    print("fast_dot...")
    A.print_repr()
    B.print_repr()
    fast_dot_product(C, A, B)
    




    

    
    output.print_repr()





        

    