import numpy as np

class layer():
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.output_size = None
        self.input_size = None

    def forwardPass(self, inputs):
        return None

    def backwardPass(self, lr, error):
        return None


class loss:
    def __init__(self):
        self.output = 0

    def run(self, y, y_hat):
        return None

    def derv(self,y, y_hat):
        return None