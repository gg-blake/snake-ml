from algorithm import vectorize, parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from python import Python
from population import dtype

alias nelts = simdwidthof[dtype]() * 2

@value
struct Matrix[rows: Int, cols: Int](Hashable):
    var data: DTypePointer[dtype]

    fn __init__(inout self):
        self.data = DTypePointer[dtype].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    fn __init__(inout self, owned data: DTypePointer[dtype]):
        self.data = data

    fn __hash__(self) -> Int:
        return self.data.__int__()

    fn __mul__(borrowed self, borrowed other: Matrix) -> Matrix[self.rows, other.cols]:
        return Matrix[self.rows, other.cols].fast_dot_product(self, other)

    fn __mul__(borrowed self, borrowed other: Float32) -> Self:
        return Matrix[self.rows, self.cols].scale(self, other)

    fn __add__(borrowed self, borrowed other: Self) -> Self:
        return Matrix[self.rows, self.cols].fast_add(self, other)

    fn average(self, other: Self) -> Self:
        return self + other * 0.5

    fn print_repr(self):
        for i in range(self.rows):
            for j in range(self.cols):
                var item = self.__getitem__(i, j)
                print(item)

    @staticmethod
    fn encode(array: VariadicList[Int]) -> Matrix[rows, cols]:
        var m = Matrix[rows, cols]()
        for index in range(len(array)):
            m.__setitem__(1, index, array[index])

        return m

    fn decode(self) -> DynamicVector[Int]:
        if rows > 1:
            return DynamicVector[Int]()

        var m = DynamicVector[Int]()
        for index in range(cols):
            m.__setitem__(index, self.__getitem__(0, index).to_int())

        return m


    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[dtype].alloc(rows * cols)
        random.rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> SIMD[dtype, 1]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, value: SIMD[dtype, 1]):
        self.store[1](y, x, value)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[dtype, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, value: SIMD[dtype, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, value)

    # Perform 2D tiling on the iteration space defined by end_x and end_y.
    @staticmethod
    fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](borrowed end_x: Int, borrowed end_y: Int):
        # Note: this assumes that ends are multiples of the tiles.
        for y in range(0, end_y, tile_y):
            for x in range(0, end_x, tile_x):
                tiled_fn[tile_x, tile_y](x, y)

    # Unroll the vectorized loop by a constant factor.
    @staticmethod
    fn fast_dot_product(borrowed matrix_a: Matrix, borrowed matrix_b: Matrix) -> Matrix[matrix_a.rows, matrix_b.cols]:
        var matrix_product = Matrix[matrix_a.rows, matrix_b.cols]()
        @parameter
        fn calc_row(m: Int):
            @parameter
            fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
                for k in range(y, y + tile_y):
                    @parameter
                    fn dot[nelts: Int](n: Int):
                        matrix_product.store(m, n + x, matrix_product.load[nelts](m, n + x) + matrix_a[m, k] * matrix_b.load[nelts](k, n + x))

                    # Vectorize by nelts and unroll by tile_x/nelts
                    # Here unroll factor is 4
                    alias unroll_factor = tile_x // nelts
                    vectorize[dot, nelts, tile_x, unroll_factor]()

            alias tile_size = 4
            Matrix[matrix_a.rows, matrix_b.cols].tile[calc_tile, nelts * tile_size, tile_size](matrix_a.cols, matrix_a.cols)

        parallelize[calc_row](matrix_product.rows, matrix_product.rows)
        return matrix_product
    
    @staticmethod
    fn fast_add(borrowed matrix_a: Matrix, borrowed matrix_b: Matrix) -> Matrix[matrix_a.rows, matrix_a.cols]:
        var matrix_sum = Matrix[matrix_a.rows, matrix_a.cols]()

        @parameter
        fn calc_add(m: Int):
            for n in range(matrix_sum.cols):
                matrix_sum.store[nelts](m, n, matrix_a.load[nelts](m, n) + matrix_b.load[nelts](m, n))

        parallelize[calc_add](matrix_sum.rows, matrix_sum.rows)

        return matrix_sum

    @staticmethod
    fn fast_sigmoid(borrowed matrix: Matrix) -> Matrix[matrix.rows, matrix.cols]:
        var matrix_output = Matrix[matrix.rows, matrix.cols]()

        @parameter
        fn calc_sigmoid(m: Int):
            matrix_output.store[nelts](m, 1, 1 / (1 + math.exp(-matrix[m, 1])))

        parallelize[calc_sigmoid](matrix_output.rows, matrix_output.rows)

        return matrix_output

    @staticmethod
    fn mutate[dtype: DType](borrowed matrix: Matrix, borrowed mutation_rate: Float64) -> Matrix[matrix.rows, matrix.cols]:
        var matrix_mutated = Matrix[matrix.rows, matrix.cols]()

        random.randn(matrix_mutated.data, 0, 0.1)

        @parameter
        fn mutate_row(m: Int):
            for n in range(matrix.cols):
                @parameter
                fn mutate_cell[nelts: Int](n: Int):
                    if random.random_float64() < mutation_rate:
                        matrix.store(m, n, matrix.load[nelts](m, n) + matrix_mutated.load[nelts](m, n))
                    else:
                        matrix.store(m, n, matrix.load[nelts](m, n))

                vectorize[mutate_cell, nelts](matrix.cols)

        parallelize[mutate_row](matrix.rows, matrix.rows)

        return matrix_mutated

    @staticmethod
    fn fast_transpose[dtype: DType](borrowed matrix: Matrix) -> Matrix[matrix.rows, matrix.cols]:
        var matrix_transposed = Matrix[matrix.rows, matrix.cols]()

        @parameter
        fn transpose(m: Int):
            for n in range(matrix.cols):
                matrix_transposed.store(n, m, matrix.load[nelts](m, n))

        parallelize[transpose](matrix.rows, matrix.rows)

        return matrix_transposed

    @staticmethod
    fn identity_matrix[size: Int](borrowed scale: Float32 = 1.0) -> Matrix[size, size]:
        var matrix = Matrix[size, size]()
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                if i == j:
                    matrix.store[nelts](i, j, scale)

        return matrix

    @staticmethod
    fn scale(borrowed matrix: Matrix, borrowed scale: Float32) -> Matrix[matrix.rows, matrix.cols]:
        var scaled_identity = Matrix[matrix.rows, matrix.cols].identity_matrix[matrix.cols](scale)
        return Matrix[matrix.rows, matrix.cols].fast_dot_product(matrix, scaled_identity)

    

