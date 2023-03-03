import numpy as np


class relu:
    def __init__(self):
        self.out = None

    def run(self, z):
        return np.maximum(0, z)

    def derv(self, z):
        return (z > 0) + 1


class sigmoid:
    def __init__(self):
        self.output = None
        self.input = None

    def run(self, z):
        self.input = z
        return abs(1/(1+np.exp(-z)))

    def derv(self, error):
        z = self.input
        return (self.run(z)*(1-self.run(z)))*error


class tanh:
    def __init__(self):
        self.input = None
        self.output = None

    def run(self, z):
        self.input = z
        self.output = np.tanh(z)
        return self.output

    def derv(self,error):
        z = self.input
        return (1-np.tanh(z)**2) * error
