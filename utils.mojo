struct RGB:
    var r: Int
    var g: Int
    var b: Int

    fn __init__(inout self, r: Int, g: Int, b: Int):
        self.r = r
        self.g = g
        self.b = b

    fn __add__(self, other: Self) -> RGB:
        return RGB((self.r + other.r) // 2, (self.g + other.g) // 2, (self.b + other.b) // 2)
    
    fn __str__(self):
        return f"{self.to_hex()}"
    
    fn __repr__(self):
        return self.__str__()

    fn to_hex(self):
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    fn set_value(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    
if __name__ == "__main__":
    print(RGB(180, 90, 25).to_hex())