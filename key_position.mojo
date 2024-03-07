from python import Python

@value
struct KeyPosition(KeyElement):
    var x: Int
    var y: Int

    fn __init__(inout self, x: Int, y: Int):
        self.x = x
        self.y = y

    fn __hash__(self) -> Int:
        try:
            var hashlib = Python.import_module("hashlib")
            var md5_hash = hashlib.md5()
            md5_hash.update((self.x, self.y))
            return md5_hash.hexdigest().to_float64().to_int()
        except:
            var hashlib = Python.none()
            return 0

    fn __eq__(self, other: Self) -> Bool:
        return self.x == other.x and self.y == other.y