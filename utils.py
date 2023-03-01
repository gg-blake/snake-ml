class RGB:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def __add__(self, other):
        return RGB((self.r + other.r) // 2, (self.g + other.g) // 2, (self.b + other.b) // 2)
    
    def __str__(self):
        return f"{self.to_hex()}"
    
    def __repr__(self):
        return self.__str__()

    def to_hex(self):
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    def set_value(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    
if __name__ == "__main__":
    print(RGB(180, 90, 25).to_hex())