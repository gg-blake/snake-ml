from population import dtype
from tensor import Tensor, TensorShape
from random import rand
from utils.index import Index
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from algorithm import vectorize, parallelize


alias nelts = simdwidthof[dtype]()

@value
struct Tensor2D[rows: Int, cols: Int]:
    var data: DTypePointer[dtype]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = DTypePointer[dtype].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, owned data: DTypePointer[dtype]):
        self.data = data

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[dtype].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> SIMD[dtype, 1]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: SIMD[dtype, 1]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[dtype, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[dtype, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)

    fn __str__(self):
        var tensor = Tensor[dtype](self.data, TensorShape(self.rows, self.cols))
        print(tensor)

fn matmul_parallelized(C: Tensor2D, A: Tensor2D, B: Tensor2D):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[dot, nelts, size = C.cols]()
    parallelize[calc_row](C.rows, C.rows)

fn main():
    var tensor_a = Tensor2D[3, 4]()
    tensor_a[0, 0] = 4
    tensor_a[1, 0] = 3
    tensor_a[2, 0] = 5
    tensor_a[0, 1] = 2
    tensor_a[1, 1] = 3
    tensor_a[2, 1] = 5
    tensor_a[0, 2] = 8
    tensor_a[1, 2] = 1
    tensor_a[2, 2] = 2
    tensor_a[0, 3] = 3
    tensor_a[1, 3] = 3
    tensor_a[2, 3] = 6
    var tensor_b = Tensor2D[4, 3]()
    tensor_b[0, 0] = 4
    tensor_b[0, 1] = 3
    tensor_b[0, 2] = 5
    tensor_b[1, 0] = 2
    tensor_b[1, 1] = 3
    tensor_b[1, 2] = 5
    tensor_b[2, 0] = 8
    tensor_b[2, 1] = 1
    tensor_b[2, 2] = 2
    tensor_b[3, 0] = 3
    tensor_b[3, 1] = 3
    tensor_b[3, 2] = 6
    var tensor_c = Tensor2D[3, 3]()
    print(tensor_a.__str__())
    print(tensor_b.__str__())
    print(tensor_c.__str__())
    matmul_parallelized(tensor_c, tensor_a, tensor_b)
    print(tensor_c.__str__())
    tensor_a.data.free()
    tensor_b.data.free()
    tensor_c.data.free()

