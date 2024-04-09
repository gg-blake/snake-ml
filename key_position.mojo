from python import Python
from collections import Set
from math import sqrt

alias dtype = DType.float32

@value
struct Position(KeyElement):
    var x: SIMD[dtype, 1]
    var y: SIMD[dtype, 1]
    var data: SIMD[dtype, 2]
    var hash: Int

    fn __init__(inout self, x: SIMD[dtype, 1], y: SIMD[dtype, 1]):
        self.x = x
        self.y = y
        self.data = SIMD[dtype, 2](x, y)
        self.hash = hash(self.data)

    fn __hash__(self) -> Int:
        return self.hash

    fn __eq__(self, other: Self) -> Bool:
        return self.hash == other.hash

    fn __ne__(self, other: Self) -> Bool:
        return self.hash != other.hash

    fn __str__(self) -> String:
        return str(self.data)

    fn __getitem__(self, index: Int) raises -> SIMD[dtype, 1]:
        return self.data[index]

    fn __setitem__(inout self, index: Int, value: SIMD[dtype, 1]) raises:
        self.data[index] = value

    fn __len__(self) -> Int:
        return len(self.data)

    # Calculates the euclidean distance
    fn distance(self: Self, other: Self) raises -> SIMD[dtype, 1]:
        if len(self) != 2 or len(other) != 2:
            raise "Both KeyTuples must be of size 2 to calculate euclidean distance"

        return sqrt((self[0] - other[0])**2 + (self[1] - other[1])**2)

fn print_set(s: Set[Position]):
    print("{", end="")
    for ref in s:
        print(ref[], end="")
    print("}")

fn main() raises:
    var test = Set[Position]()
    var point_a = Position(2, 4)
    var point_b = Position(6, 12)
    var point_c = Position(2, 4)
    print(point_a == point_c)
    test.add(point_a)
    test.add(point_b)
    test.add(point_c)
    print(point_a.distance(point_b))

    