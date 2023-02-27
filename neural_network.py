import numpy as np
import random

class NeuralNetwork:
    _count = 0 # Total number of NeuralNetwork instances

    def __init__(self, input_nodes, hidden_nodes, output_nodes, mother=0, father=0):
        # Instance Variables #
        self.input_nodes = input_nodes # Number of nodes in input layer
        self.hidden_nodes = hidden_nodes # Number of nodes in hidden layer
        self.output_nodes = output_nodes # Number of nodes in output layer
        self.id = NeuralNetwork._count # Unique NeuralNetwork object identifier
        self.mother = mother # NeuralNetwork id of mother
        self.father = father # NeuralNetwork id of father
        self.weights_ih = np.random.rand(self.hidden_nodes, self.input_nodes) # Weights from input layer to hidden layer
        self.weights_ho = np.random.rand(self.output_nodes, self.hidden_nodes) # Weights from hidden layer to output layer
        self.bias_h = np.random.rand(self.hidden_nodes, 1) # Biases of hidden layer
        self.bias_o = np.random.rand(self.output_nodes, 1) # Biases of output layer

        # Instance Callback #
        NeuralNetwork._count += 1 # Increment total number of NeuralNetwork objects by 1

    # String representation
    def __str__(self):
        return f"weights_ih: {self.weights_ih}, weights_ho: {self.weights_ho}, bias_h: {self.bias_h}, bias_o: {self.bias_o}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()
    
    # Dictionary representation
    def __dict__(self):
        # This function enables models to be saved as JSON files
        return {
            "input_nodes": self.input_nodes,
            "hidden_nodes": self.hidden_nodes,
            "output_nodes": self.output_nodes,
            "weights_ih": self.weights_ih.tolist(),
            "weights_ho": self.weights_ho.tolist(),
            "bias_h": self.bias_h.tolist(),
            "bias_o": self.bias_o.tolist()
        }

    # Perform a feedforward operation on this NeuralNetwork object and return the output
    def feed(self, input_array):
        # Convert input array to numpy array
        inputs = np.array(input_array, ndmin=2).T

        # Feed input layer to hidden layer
        hidden = np.dot(self.weights_ih, inputs)
        hidden = np.add(hidden, self.bias_h)
        hidden = self.sigmoid(hidden)

        # Feed hidden layer to output layer
        output = np.dot(self.weights_ho, hidden)
        output = np.add(output, self.bias_o)
        output = self.sigmoid(output)

        return output
    
    # Determines the activation of a node
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Potentially mutate the weights and biases of this NeuralNetwork object
    def mutate(self, mutation_rate):
        # Vectorized function to mutate a value
        def mutate(val):
            # Mutate value if random number is less than mutation rate
            if random.random() < mutation_rate:
                return val + random.gauss(0, 0.1)
            else:
                return val
            
        # Vectorize the mutate function
        mutate = np.vectorize(mutate)

        # Set weights and biases to new mutated numpy array
        self.weights_ih = mutate(self.weights_ih)
        self.weights_ho = mutate(self.weights_ho)
        self.bias_h = mutate(self.bias_h)
        self.bias_o = mutate(self.bias_o)

    # Returns a copy of this NeuralNetwork object
    def clone(self):
        # Create new NeuralNetwork object with identical instance variables
        nn = NeuralNetwork(self.input_nodes, self.hidden_nodes, self.output_nodes)
        # Copy the weights and biases
        nn.weights_ih = np.copy(self.weights_ih)
        nn.weights_ho = np.copy(self.weights_ho)
        nn.bias_h = np.copy(self.bias_h)
        nn.bias_o = np.copy(self.bias_o)
        nn.mother, nn.father = self.id, self.id
        return nn