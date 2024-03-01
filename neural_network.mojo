from python import Python
from collections.vector import DynamicVector
from tensor import TensorShape, tensor
from algorithm.functional import vectorize

struct NeuralNetwork:
    var i_size: Int
    var h_size: Int
    var o_size: Int
    var ih: Tensor[DType.float16]
    var ho: Tensor[DType.float16]
    var h: Tensor[DType.float16]
    var o: Tensor[DType.float16]
    
    fn __init__(inout self, i_size: Int, h_size: Int, o_size: Int) raises:
        let np = Python.import_module("numpy")
        self.i_size = i_size
        self.h_size = h_size
        self.o_size = o_size
        self.ih = random.rand[DType.float16](TensorShape(h_size, i_size))
        self.ho = random.rand[DType.float16](TensorShape(o_size, h_size))
        self.h = random.rand[DType.float16](TensorShape(h_size, 1))
        self.o = random.rand[DType.float16](TensorShape(o_size, 1))

    fn feed(self, borrowed input: Tensor[DType.float16]):
        var inputs: Tensor[DType.float16] = input.reshape(TensorShape(self.h_size, self.i_size)).transpose()

        var hidden: Tensor[DType.float16] = self.ih * inputs
        hidden = hidden + self.h
        hidden = self.sigmoid(hidden)

        var output: Tensor[DType.float16] = self.ho * hidden
        output = output + self.o
        output = self.sigmoid(output)

        return output

    # Determines the activation of a node
    fn sigmoid(self, t: Tensor) raises -> Tensor:
        return Tensor.ones(t.data.size).divide(
            t.multiply(-1.0).exponential().plus(1.0)
        )

    fn mutate(self, mutation_rate: Float16) raises -> Float64:
        let rand = Python.import_module("random")
        let np = Python.import_module("numpy")
        fn mutate(value: Float64) raises:
            if rand.random() < mutation_rate:
                return value + rand.gauss(0, 0.1).to_float64
            else:
                return value

        var mutate = vectorize(mutate)
        
'''

