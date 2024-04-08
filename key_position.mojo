from python import Python
from collections import Set
from math import sqrt

alias dtypesize = simdwidthof[DType.float64]() * 2

@value
struct KeyTuple(KeyElement):
    var data: SIMD[DType.float64, 2]
    var hash: Int

    fn __init__(inout self, x: Float64, y: Float64):
        self.data = SIMD[DType.float64, 2](x, y)
        self.hash = self.data.__hash__()
        

    fn __hash__(self) -> Int:
        return self.hash

    fn __eq__(self, other: Self) -> Bool:
        return self.hash == other.hash

    fn __ne__(self, other: Self) -> Bool:
        return not self == other 

    fn __str__(self) -> String:
        return self.data

    fn __getitem__(self, index: Int) raises -> Float64:
        return self.data[index]

    fn __setitem__(inout self, index: Int, value: Float64) raises:
        self.data[index] = value

    fn __len__(self) -> Int:
        return len(self.data)

    # Calculates the euclidean distance
    fn distance(self: Self, other: Self) raises -> Float64:
        if len(self) != 2 or len(other) != 2:
            raise "Both KeyTuples must be of size 2 to calculate euclidean distance"

        return sqrt((self[0] - other[0])**2 + (self[1] - other[1])**2)

fn print_set(s: Set[Int]):
    print("{", end="")
    for ref in s:
        print(ref[], end="")
    print("}")

fn main() raises:
    var test = Set[Int]()
    var point_a = KeyTuple(x=2, y=4)
    var point_b = KeyTuple(x=3, y=4)
    var point_c = KeyTuple(x=2, y=4)
    var a: PythonObject = [4, 5, 6]
    print(a)
    test.add(point_a.hash)
    print_set(test)
    test.add(point_b.hash)
    print_set(test)
    test.add(point_c.hash)
    print_set(test)
    