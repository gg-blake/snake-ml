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

    fn feed(self, input: Tensor[DType.float16]) raises -> Tensor[DType.float16]:
        var new_tensor = input

        var inputs: Tensor[DType.float16] = Tensor.reshape(new_tensor, TensorShape(self.h_size, self.i_size))

        var hidden: Tensor[DType.float16] = self.ih * inputs
        hidden = hidden + self.h
        hidden = self.sigmoid(hidden)

        var output: Tensor[DType.float16] = self.ho * hidden
        output = output + self.o
        output = self.sigmoid(output)

        return output

    # Determines the activation of a node
    fn sigmoid(self, t: Tensor[DType.float16]) raises -> Tensor[DType.float16]:
        '''var math = Python.import_module("math")
        return t.__truediv__(
            math.e**(t.__mul__(-1.0)).__add__(1.0)
        )'''
        return t

    fn mutate(self, mutation_rate: Float16) raises -> Float16:
        let rand = Python.import_module("random")
        let np = Python.import_module("numpy")
        '''fn v_mutate(value: Float16) raises -> Float16:
            if rand.random() < mutation_rate:
                return value + rand.gauss(0, 0.1).to_float16
            else:
                return value

        var mutate = vectorize(v_mutate)'''

        return 1.0
        

