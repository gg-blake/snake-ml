from python import Python
from population import dtype, Population
from random import rand
from tensor import TensorShape, Tensor



@value
struct NeuralNetwork[input_size: Int, hidden_size: Int, output_size: Int](Hashable):
    var id: Int
    var input_to_hidden_weights: PythonObject
    var hidden_to_output_weights: PythonObject
    var hidden_node_biases: PythonObject
    var output_node_biases: PythonObject
    
    fn __init__(inout self, id: Int):
        try:
            var torch = Python.import_module("torch")
            if torch.cuda.is_available():
                var device = torch.device("cuda")
            else:
                var device = torch.device("cpu")

            self.input_to_hidden_weights = torch.rand(hidden_size, input_size)
            self.hidden_to_output_weights = torch.rand(output_size, hidden_size)
            self.hidden_node_biases = torch.rand(hidden_size, 1)
            self.output_node_biases = torch.rand(output_size, 1)
            self.id = id
        except:
            var torch = Python.none()
            self.input_to_hidden_weights = Python.none()
            self.hidden_to_output_weights = Python.none()
            self.hidden_node_biases = Python.none()
            self.output_node_biases = Python.none()
            self.id = id

    fn __init__(inout self, input_to_hidden_weights: PythonObject, hidden_to_output_weights: PythonObject, hidden_node_biases: PythonObject, output_node_biases: PythonObject, id: Int):
        self.id = id
        self.input_to_hidden_weights = input_to_hidden_weights
        self.hidden_to_output_weights = hidden_to_output_weights
        self.hidden_node_biases = hidden_node_biases
        self.output_node_biases = output_node_biases

    fn __copyinit__(inout self, existing: Self):
        self.id = existing.id
        self.input_to_hidden_weights = existing.input_to_hidden_weights
        self.hidden_to_output_weights = existing.hidden_to_output_weights
        self.hidden_node_biases = existing.hidden_node_biases
        self.output_node_biases = existing.output_node_biases

    fn __hash__(self) -> Int:
        return self.id

    fn feed(self, input_array: PythonObject) raises -> PythonObject:
        var torch = Python.import_module("torch")
        var hidden_array = torch.mm(self.input_to_hidden_weights, input_array) + self.hidden_node_biases
        var output_array = torch.mm(self.hidden_to_output_weights, hidden_array) + self.output_node_biases
        return output_array

    fn blend_genetic_traits(self: Self, other: Self, id: Int) raises -> Self:
        var avg_matrix_ih = (self.input_to_hidden_weights + other.input_to_hidden_weights) * 0.5
        var avg_matrix_ho = (self.hidden_to_output_weights + other.hidden_to_output_weights) * 0.5
        var avg_matrix_h = (self.hidden_node_biases + other.hidden_node_biases) * 0.5
        var avg_matrix_o = (self.output_node_biases + other.output_node_biases) * 0.5
        # TODO: Mutate the weights and biases before return
        return Self(avg_matrix_ih, avg_matrix_ho, avg_matrix_h, avg_matrix_o, id)
        
