from tensor import Tensor, TensorShape
from utils.index import Index
from math import clamp, sqrt
from population import DTYPE, Vec3, Vec1

alias VecIndex = StaticIntTuple[1]
alias EAST = Vec3(TensorShape(3), 1.0, 0.0, 0.0)
alias WEST = Vec3(TensorShape(3), -1.0, 0.0, 0.0)
alias NORTH = Vec3(TensorShape(3), 0.0, 1.0, 0.0)
alias SOUTH = Vec3(TensorShape(3), 0.0, -1.0, 0.0)
alias ZENITH = Vec3(TensorShape(3), 0.0, 0.0, 1.0)
alias NADIR = Vec3(TensorShape(3), 0.0, 0.0, -1.0)

struct GameObject3D(CollectionElement):
    var position: Vec3
    var velocity: Dict[String, Vec3]

    fn __init__(inout self, x: Scalar[DTYPE], y: Scalar[DTYPE], z: Scalar[DTYPE]):
        self.position = Vec3(TensorShape(3), x, y, z)
        self.velocity = Dict[String, Vec3]()
        self.velocity["front"] = EAST
        self.velocity["back"] = WEST
        self.velocity["right"] = NORTH
        self.velocity["left"] = SOUTH
        self.velocity["up"] = ZENITH
        self.velocity["down"] = NADIR

    fn __init__(inout self, position: Vec3, velocity: Dict[String, Vec3]):
        self.position = position
        self.velocity = velocity

    fn __getitem__(inout self, key: String) raises -> Vec3:
        return self.velocity[key]

    fn __getitem__(self, index: Int) -> Scalar[DTYPE]:
        return self.position[index]

    fn __eq__(self, pos: Self) -> Bool:
        return self[0] == pos[0] and self[1] == pos[1] and self[2] == pos[2]

    fn __add__(self, other: Self) raises -> Self:
        return GameObject3D(self.position + other.position, self.velocity)

    fn __copyinit__(inout self, existing: Self):
        self.position = existing.position
        self.velocity = existing.velocity

    fn __moveinit__(inout self, owned existing: Self):
        self.position = existing.position^
        self.velocity = existing.velocity^

    fn reset(inout self):
        self.position = Vec3(TensorShape(3), 0.0, 0.0, 0.0)
        self.velocity["front"] = EAST
        self.velocity["back"] = WEST
        self.velocity["right"] = NORTH
        self.velocity["left"] = SOUTH
        self.velocity["up"] = ZENITH
        self.velocity["down"] = NADIR

    fn turn_left(inout self) raises:
        # Shift relative axis velocity to the left
        var temp = self.velocity["front"]
        self.velocity["front"] = self.velocity["left"]
        self.velocity["left"] = self.velocity["back"]
        self.velocity["back"] = self.velocity["right"]
        self.velocity["right"] = temp

    fn turn_right(inout self) raises:
        var temp = self.velocity["front"]
        self.velocity["front"] = self.velocity["right"]
        self.velocity["right"] = self.velocity["back"]
        self.velocity["back"] = self.velocity["left"]
        self.velocity["left"] = temp

    fn turn_up(inout self) raises:
        var temp = self.velocity["front"]
        self.velocity["front"] = self.velocity["up"]
        self.velocity["up"] = self.velocity["back"]
        self.velocity["back"] = self.velocity["down"]
        self.velocity["down"] = temp

    fn turn_down(inout self) raises:
        var temp = self.velocity["front"]
        self.velocity["front"] = self.velocity["down"]
        self.velocity["down"] = self.velocity["back"]
        self.velocity["back"] = self.velocity["up"]
        self.velocity["up"] = temp

    fn move_forward(inout self) raises:
        self.position = self.position + self.velocity["front"]

    fn _cmp_on_axis(self, other: Self, inout axis: Vec3) raises -> Int:
        var index = Index(self._get_eigenindex(axis))
        var direction: Scalar[DTYPE] = axis[index]
        var delta_lr = self.position[index] - other.position[index]
        if clamp[DTYPE](delta_lr, -1, 1) * direction < 0:
            # Left
            return -1
        elif clamp[DTYPE](delta_lr, -1, 1) * direction > 0:
            # Right
            return 1
        else:
            # Equal
            return 0

    fn is_above(inout self, other: Self) raises -> Bool:
        return self._cmp_on_axis(other, self.velocity["up"]) < 0

    fn is_above_strict(inout self, other: Self) raises -> Bool:
        var cmp_lr = self._cmp_on_axis(other, self.velocity["left"])
        var cmp_fb = self._cmp_on_axis(other, self.velocity["front"])
        return self._cmp_on_axis(other, self.velocity["up"]) & ~cmp_lr & ~cmp_fb < 0

    fn is_below(inout self, other: Self) raises -> Bool:
        return self._cmp_on_axis(other, self.velocity["down"]) < 0

    fn is_below_strict(inout self, other: Self) raises -> Bool:
        var cmp_lr = self._cmp_on_axis(other, self.velocity["left"])
        var cmp_fb = self._cmp_on_axis(other, self.velocity["front"])
        return self._cmp_on_axis(other, self.velocity["down"]) & ~cmp_lr & ~cmp_fb < 0

    fn is_left(inout self, other: Self) raises -> Bool:
        return self._cmp_on_axis(other, self.velocity["left"]) < 0

    fn is_left_strict(inout self, other: Self) raises -> Bool:
        var cmp_lr = self._cmp_on_axis(other, self.velocity["left"])
        var cmp_fb = self._cmp_on_axis(other, self.velocity["front"])
        return self._cmp_on_axis(other, self.velocity["left"]) & ~cmp_lr & ~cmp_fb < 0

    fn is_right(inout self, other: Self) raises -> Bool:
        return self._cmp_on_axis(other, self.velocity["right"]) < 0

    fn is_right_strict(inout self, other: Self) raises -> Bool:
        var cmp_lr = self._cmp_on_axis(other, self.velocity["left"])
        var cmp_fb = self._cmp_on_axis(other, self.velocity["front"])
        return self._cmp_on_axis(other, self.velocity["right"]) & ~cmp_lr & ~cmp_fb < 0

    fn is_ahead(inout self, other: Self) raises -> Bool:
        return self._cmp_on_axis(other, self.velocity["front"]) < 0

    fn is_ahead_strict(inout self, other: Self) raises -> Bool:
        var cmp_lr = self._cmp_on_axis(other, self.velocity["left"])
        var cmp_fb = self._cmp_on_axis(other, self.velocity["front"])
        return self._cmp_on_axis(other, self.velocity["front"]) & ~cmp_lr & ~cmp_fb < 0

    fn is_behind(inout self, other: Self) raises -> Bool:
        return self._cmp_on_axis(other, self.velocity["back"]) < 0

    fn is_behind_strict(inout self, other: Self) raises -> Bool:
        var cmp_lr = self._cmp_on_axis(other, self.velocity["left"])
        var cmp_fb = self._cmp_on_axis(other, self.velocity["front"])
        return self._cmp_on_axis(other, self.velocity["back"]) & ~cmp_lr & ~cmp_fb < 0

    fn _get_eigenindex(self, inout vec: Vec3) raises -> Scalar[DType.index]:
        var abs_vec = (vec.reshape(TensorShape(1, 3)))**2
        var argmax = abs_vec.argmax()
        return argmax[Index(0)]

    fn distance(self, other: Vec3) raises -> Vec1:
        var delta = (self.position - other)**2
        var sum = delta[0] + delta[1] + delta[2]
        return sqrt(sum)

    fn distance(self, other: Self) raises -> Vec1:
        return self.distance(other.position)

