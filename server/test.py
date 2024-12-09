from torch import tensor

def main(a: list[int], b: list[int]):
    tensor_A = tensor(a)
    tensor_B = tensor(b)
    return tensor_A @ tensor_B
