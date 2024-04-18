from python import Python
from tensor import Tensor

@value
struct NeuralNetwork[dtype: DType](Hashable):
    var spec: List[Int]
    var data: AnyPointer[PythonObject]
    var hash: Int
    
    fn __init__(inout self, spec: List[Int]) raises:
        var torch = Python.import_module("torch")
        self.spec = spec
        self.data = AnyPointer[PythonObject].alloc((len(spec) - 1) * 2)
        self.hash = int(self.data)
        for i in range(0, len(spec) - 1):
            self.data[2*i] = torch.rand(spec[i+1], spec[i])
            self.data[2*i+1] = torch.rand(spec[i+1], 1)

    fn __init__(inout self, spec: List[Int], owned data: AnyPointer[PythonObject]):
        self.spec = spec
        self.data = data
        self.hash = int(self.data)

    fn __moveinit__(inout self, owned existing: Self):
        self = Self(existing.spec, existing.data)

    fn __hash__(self) -> Int:
        return self.hash

    fn __str__(self) -> String:
        var result: String = "NeuralNetwork(\n"
        for i in range(len(self.spec) - 1):
            result += str(self.data[2*i]) + "\n"
            result += str(self.data[2*i+1]) + "\n"
        result += ")"
        return result

    fn __del__(owned self):
        self.data.free()

    fn __getitem__(self, index: Int) -> PythonObject:
        return self.data[index]

    fn __setitem__(inout self, index: Int, value: PythonObject):
        self.data[index] = value

    fn feed(self, input_array: PythonObject) raises -> PythonObject:
        if len(input_array) != self.spec[0]:
            raise Error("Input tensor has wrong dimensions")

        var torch = Python.import_module("torch")
        var output_array = input_array
        for i in range(len(self.spec) - 1):
            output_array = torch.special.expit(torch.mm(self.data[2*i], output_array) + self.data[2*i+1])
        return output_array

    fn mutate(inout self, strength: Float32) raises:
        var torch = Python.import_module("torch")
        for i in range(len(self.spec) - 1):
            self[2*i] = self[2*i] + strength * torch.randn_like(self[2*i])
            self[2*i+1] = self[2*i+1] + strength * torch.randn_like(self[2*i+1])

    fn average(inout self: Self, other: Self) raises:
        for i in range(len(self.spec) - 1):
            self[2*i] = (self[2*i] + other[2*i]) * 0.5
            self[2*i+1] = (self[2*i+1] + other[2*i+1]) * 0.5

    fn export(borrowed self, index: Int) raises -> Tensor[dtype]:
        if index > len(self.spec) * 2 or index < 0:
            raise Error("Index out of range")
            
        var result_weight = Tensor[dtype](index+1, index)
        var result_bias = Tensor[dtype](index, 1)
        '''for i in range(rows):
            for j in range(cols):
                var map_index = StaticIntTuple[2](i, j)
                var data = SIMD[DType.float64, 1](self.data[index][i][j].item().to_float64()).cast[dtype]()
                result.__setitem__(map_index, data)'''

        return result_weight

    fn export_all(self) raises -> List[Tensor[dtype]]:
        var tensor_list = List[Tensor[dtype]]()
        for i in range(len(self.spec)):
            tensor_list.append(self.export(i))

        return tensor_list

        


fn main() raises:
    var torch = Python.import_module("torch")
    var test_a = NeuralNetwork[DType.float32](List[Int](20, 20, 4))
    print(test_a.export(2))
    
    