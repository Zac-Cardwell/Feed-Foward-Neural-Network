import numpy as np
import FeedFowardNetwork as FCL
import activationFunctions as af
import lossFunction as lf
import shells


class model():

    def __init__(self):
        self.layers = []
        self.loss_func = None

    def add_fclayer(self, output_size, act_fun, input_size=1):
        if len(self.layers) > 0:
            input_size = self.layers[-1].output_size
        self.layers.append(FCL.FCLayer(input_size, output_size, act_fun))

    def fpass(self, data):
        results = []
        for i in data:
            output = np.array([i])
            for j in self.layers:
                output = j.forwardPass(output)
            results.append(output)
        return np.array(results)

    def fit(self, x_data, y_data, batch_size,  epochs, lr, loss_func):
        for k in range(epochs):
            error = 0
            for l in range(len(x_data)):
                y_hat = x_data[l]
                for m in self.layers:
                    y_hat = m.forwardPass(y_hat)
                error += loss_func.run(y_data[l], y_hat)
                err = loss_func.derv(y_data[l], y_hat)
                for layer in reversed(self.layers):
                    err = layer.backwardPass(err, lr)
            print(f'Epoch: {k+1}  Error:{error/len(x_data)}')

    def summary(self):
        print(f'\nNumber of layers: {len(self.layers)}\n')
        for i in range(len(self.layers)):
            print(f'Layer:{i+1} input:{self.layers[i].input_size} output:{self.layers[i].output_size} '
                  f'act:{self.layers[i].act_func}\n')


x = np.array([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]])
y = np.array([[[0]], [[1]], [[0]], [[1]]])
test = model()
test.add_fclayer(3, af.tanh(), 2)
test.add_fclayer(1, af.tanh())
test.fit(x, y, 2, epochs=1000, lr=.1, loss_func=lf.MSE())
print(test.fpass(x))


