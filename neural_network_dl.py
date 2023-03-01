import numpy as np
import random
from .utils import RGB

class DoubleLayerNeuralNetwork:
    _count = 0 # Total number of DoubleLayerNeuralNetwork instances

    def __init__(self, input_nodes, first_hidden_nodes, second_hidden_nodes, output_nodes, mother=RGB(-1,-1,-1), father=RGB(-1,-1,-1)):
        # Instance Variables #
        self.input_nodes = input_nodes # Number of nodes in input layer
        self.first_hidden_nodes = first_hidden_nodes # Number of nodes in first hidden layer
        self.second_hidden_nodes = second_hidden_nodes # Number of nodes in second hidden layer
        self.output_nodes = output_nodes # Number of nodes in output layer
        if mother.r == -1:
            self.id = RGB(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            self.id = random.choice([mother, father])
        self.mother = mother # DoubleLayerNeuralNetwork id of mother
        self.father = father # DoubleLayerNeuralNetwork id of father
        self.weights_ih1 = np.random.rand(self.first_hidden_nodes, self.input_nodes) # Weights from input layer to hidden layer
        self.weights_h1h2 = np.random.rand(self.second_hidden_nodes, self.first_hidden_nodes) # Weights from hidden layer to hidden layer
        self.weights_h2o = np.random.rand(self.output_nodes, self.second_hidden_nodes) # Weights from hidden layer to output layer
        self.bias_h1 = np.random.rand(self.first_hidden_nodes, 1) # Biases of second hidden layer
        self.bias_h2 = np.random.rand(self.second_hidden_nodes, 1) # Biases of second hidden layer
        self.bias_o = np.random.rand(self.output_nodes, 1) # Biases of output layer

        # Instance Callback #
        DoubleLayerNeuralNetwork._count += 1 # Increment total number of DoubleLayerNeuralNetwork objects by 1

    # String representation
    def __str__(self):
        return f"weights_ih1: {self.weights_ih1}, weights_h1h2: {self.weights_h1h2}, weights_h2o: {self.weights_h2o}, bias_h1: {self.bias_h1}, bias_h2: {self.bias_h2}, bias_o: {self.bias_o}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()
    
    # Dictionary representation
    def __dict__(self):
        # This function enables models to be saved as JSON files
        return {
            "input_nodes": self.input_nodes,
            "first_hidden_nodes": self.first_hidden_nodes,
            "second_hidden_nodes": self.second_hidden_nodes,
            "output_nodes": self.output_nodes,
            "weights_ih1": self.weights_ih1.tolist(),
            "weights_h1h2": self.weights_h1h2.tolist(),
            "weights_h2o": self.weights_h2o.tolist(),
            "bias_h1": self.bias_h1.tolist(),
            "bias_h2": self.bias_h2.tolist(),
            "bias_o": self.bias_o.tolist()
        }

    # Perform a feedforward operation on this DoubleLayerNeuralNetwork object and return the output
    def feed(self, input_array):
        # Convert input array to numpy array
        inputs = np.array(input_array, ndmin=2).T

        # Feed input layer to first hidden layer
        first_hidden = np.dot(self.weights_ih1, inputs)
        first_hidden = np.add(first_hidden, self.bias_h1)
        first_hidden = self.sigmoid(first_hidden)

        # Feed first hidden layer to second hidden layer
        second_hidden = np.dot(self.weights_h1h2, first_hidden)
        second_hidden = np.add(second_hidden, self.bias_h2)
        second_hidden = self.sigmoid(second_hidden)

        # Feed hidden layer to output layer
        output = np.dot(self.weights_h2o, second_hidden)
        output = np.add(output, self.bias_o)
        output = self.sigmoid(output)

        return output
    
    # Determines the activation of a node
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Potentially mutate the weights and biases of this DoubleLayerNeuralNetwork object
    def mutate(self, mutation_rate):
        # Vectorized function to mutate a value
        def mutate(val):
            # Mutate value if random number is less than mutation rate
            if random.random() < mutation_rate:
                if self.id:
                    # Keep the mutated id colors within the acceptable colorspace
                    self.id.r += random.randint(-20, 20)
                    self.id.g += random.randint(-20, 20)
                    self.id.b += random.randint(-20, 20)

                    if self.id.r < 0:
                        self.id.r = 0
                    elif self.id.r > 255:
                        self.id.r = 255

                    if self.id.g < 0:
                        self.id.g = 0
                    elif self.id.g > 255:
                        self.id.g = 0

                    if self.id.b < 0:
                        self.id.b = 0
                    elif self.id.b > 255:
                        self.id.b = 255
                return val + random.gauss(0, 0.1)
            else:
                return val
            
        # Vectorize the mutate function
        mutate = np.vectorize(mutate)

        # Set weights and biases to new mutated numpy array
        self.weights_ih1 = mutate(self.weights_ih1)
        self.weights_h1h2 = mutate(self.weights_h1h2)
        self.weights_h2o = mutate(self.weights_h2o)
        self.bias_h1 = mutate(self.bias_h1)
        self.bias_h2 = mutate(self.bias_h2)
        self.bias_o = mutate(self.bias_o)


    # Returns a copy of this DoubleLayerNeuralNetwork object
    def clone(self):
        # Create new DoubleLayerNeuralNetwork object with identical instance variables
        nn = DoubleLayerNeuralNetwork(self.input_nodes, self.first_hidden_nodes, self.second_hidden_nodes, self.output_nodes)
        # Copy the weights and biases
        nn.weights_ih1 = np.copy(self.weights_ih1)
        nn.weights_h1h2 = np.copy(self.weights_h1h2)
        nn.weights_h2o = np.copy(self.weights_h2o)
        nn.bias_h1 = np.copy(self.bias_h1)
        nn.bias_h2 = np.copy(self.bias_h2)
        nn.bias_o = np.copy(self.bias_o)
        nn.mother, nn.father = self.id, self.id
        return nn