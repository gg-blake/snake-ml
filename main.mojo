from python import Python
from population import Population, Snake

fn main() raises:
    let py = Python.import_module("builtins")
    let snake_count: String = py.input("Enter the number of snakes per generation:")
    