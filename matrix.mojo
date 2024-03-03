from algorithm import vectorize, parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from python import Python



struct Matrix[dtype: DType, rows: Int, cols: Int]:
    var data: DTypePointer[dtype]
    alias nelts = simdwidthof[dtype]() * 2

    fn __init__(inout self):
        self.data = DTypePointer[dtype].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    fn __init__(inout self, data: DTypePointer[dtype]):
        self.data = data

    fn print_repr(self):
        for i in range(self.rows):
            for j in range(self.cols):
                var item = self.__getitem__(i, j)
                print(item)


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
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

# Unroll the vectorized loop by a constant factor.
fn fast_dot_product[dtype: DType](C: Matrix[dtype], A: Matrix[dtype], B: Matrix[dtype]):
    alias nelts = simdwidthof[dtype]() * 2
    
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(m, n + x, C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x))

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                alias unroll_factor = tile_x // nelts
                vectorize[dot, nelts, tile_x, unroll_factor]()

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](C.rows, C.rows)
   
fn fast_add[dtype: DType](C: Matrix[dtype], A: Matrix[dtype], B: Matrix[dtype]):
    alias nelts = simdwidthof[dtype]() * 2

    @parameter
    fn calc_add(m: Int):
        for n in range(C.cols):
            C.store[nelts](m, n, A.load[nelts](m, n) + B.load[nelts](m, n))

    parallelize[calc_add](C.rows, C.rows)

fn fast_sigmoid[dtype: DType](C: Matrix[dtype], A: Matrix[dtype]):
    alias nelts = simdwidthof[dtype]() * 2

    @parameter
    fn calc_sigmoid(m: Int):
        C.store[nelts](m, 1, 1 / (1 + math.exp(-A[m, 1])))

    parallelize[calc_sigmoid](C.rows, C.rows)

fn mutate[dtype: DType](B: Matrix[dtype], A: Matrix[dtype], mutation_rate: Float64):
    alias nelts = simdwidthof[dtype]() * 2

    random.randn[dtype](B.data, 0, 0.1)

    @parameter
    fn mutate_row(m: Int):
        for n in range(A.cols):
            @parameter
            fn mutate_cell[nelts: Int](n: Int):
                if random.random_float64() < mutation_rate:
                    B.store(m, n, A.load[nelts](m, n) + B.load[nelts](m, n))
                else:
                    B.store(m, n, A.load[nelts](m, n))

            vectorize[mutate_cell, nelts](A.cols)

    parallelize[mutate_row](A.rows, A.rows)

fn fast_transpose[dtype: DType](B: Matrix[dtype], A: Matrix[dtype]):
    alias nelts = simdwidthof[dtype]() * 2

    @parameter
    fn transpose(m: Int):
        for n in range(A.cols):
            B.store(n, m, A.load[nelts](m, n))

    parallelize[transpose](A.rows, A.rows)

                    
                
