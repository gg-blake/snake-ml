from python import Python

@value
struct NeuralNetwork[input_size: Int, hidden_size: Int, output_size: Int](Hashable):
    var data: AnyPointer[PythonObject]
    var hash: Int
    
    fn __init__(inout self) raises:
        self.data = AnyPointer[PythonObject].alloc(4)
        var torch = Python.import_module("torch")
        if torch.cuda.is_available():
            var device = torch.device("cuda")
        else:
            var device = torch.device("cpu")

        
        self.data[0] = torch.rand(hidden_size, input_size) # ih_weights
        self.data[1] = torch.rand(output_size, hidden_size) # ho_weights
        self.data[2] = torch.rand(hidden_size, 1) # h_biases
        self.data[3] = torch.rand(output_size, 1) # o_biases
        self.hash = int(self.data)

    fn __init__(inout self, owned data: AnyPointer[PythonObject]):
        self.data = data
        self.hash = int(self.data)

    fn __moveinit__(inout self, owned existing: Self):
        self = Self(existing.data)

    fn __hash__(self) -> Int:
        return self.hash

    fn __str__(self) -> String:
        var result: String = "NeuralNetwork(\n"
        result += str(self.data[0]) + "\n"
        result += str(self.data[1]) + "\n"
        result += str(self.data[2]) + "\n"
        result += str(self.data[3]) + "\n"
        result += ")"
        return result

    fn __del__(owned self):
        self.data.free()

    fn __setitem__(inout self, index: Int, val: PythonObject):
        self.data[index] = val

    fn __getitem__(borrowed self, index: Int) -> PythonObject:
        return self.data[index]

    fn feed(self, input_array: PythonObject) raises -> PythonObject:
        var torch = Python.import_module("torch")
        var hidden_array = torch.special.expit(torch.mm(self.data[0], input_array) + self.data[2])
        var output_array = torch.special.expit(torch.mm(self.data[1], hidden_array) + self.data[3])
        return output_array

    fn mutate(inout self, strength: Float32) raises:
        var torch = Python.import_module("torch")
        self[0] = self[0] + strength * torch.randn_like(self[0])
        self[1] = self[1] + strength * torch.randn_like(self[1])
        self[2] = self[2] + strength * torch.randn_like(self[2])
        self[3] = self[3] + strength * torch.randn_like(self[3])

    fn average(inout self: Self, other: Self) raises:
        self[0] = (self[0] + other[0]) * 0.5
        self[1] = (self[1] + other[1]) * 0.5
        self[2] = (self[2] + other[2]) * 0.5
        self[3] = (self[3] + other[3]) * 0.5

fn main() raises:
    var torch = Python.import_module("torch")
    var test_a = NeuralNetwork[4, 16, 4]()
    print(test_a)
    var test_b = test_a^
    print(test_b)
    test_b.mutate(0.5)
    print(test_b)
    