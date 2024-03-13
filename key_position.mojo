from python import Python
from collections import Set


struct DimSet(Copyable, Movable):
    var set: Set[KeyTuple]
    var Point: PythonObject

    fn __init__(inout self):
        try:
            var collections = Python.import_module("collections")
            self.Point = collections.namedtuple('Point', ['x', 'y'])
        except:
            self.Point = Python.none()
        self.set = Set[KeyTuple]()

    fn __contains__(self, key: KeyTuple) -> Bool:
        return key in self.set

    fn __str__(self) -> String:
        var result: String = "{"
        var index = 1
        for ref in self.set:
            var item = ref[]
            if index != len(self.set):
                result = result + item.__str__() + ", "
            else:
                result = result + item.__str__() + "}"
            index += 1
        return result

    fn __copyinit__(inout self, existing: Self):
        self = Self()
        self.Point = existing.Point
        for item_ref in existing.set:
            self.set.add(item_ref[])

    fn __moveinit__(inout self, owned existing: Self):
        self = Self()
        self.Point = existing.Point
        for item_ref in existing.set:
            self.set.add(item_ref[])

    fn add(inout self, x: Int, y: Int) raises:
        self.set.add(KeyTuple(x, y, self.Point))

    fn remove(inout self, x: Int, y: Int) raises:
        self.set.remove(KeyTuple(x, y, self.Point))
    


@value
struct KeyTuple(KeyElement):
    var p: PythonObject
    var x: Int
    var y: Int

    fn __init__(inout self, x: Int, y: Int, Point: PythonObject) raises:
        self.p = Point(x, y)
        self.x = x
        self.y = y

    fn __hash__(self) -> Int:
        try:
            return self.p.__hash__().to_float64().to_int()
        except:
            return 0

    fn __eq__(self, other: Self) -> Bool:
        return self.x == other.x and self.y == other.y

    fn __repr__(self) -> String:
        return self.__str__()

    fn __str__(self) -> String:
        return "(" + str(self.x) + ", " + str(self.y) + ")"