import numpy as np
import shells


class MSE(shells.loss):
    def run(self, y, y_hat):
        self.output = np.power((y - y_hat), 2)
        return self.output

    def derv(self, y, y_hat):
        return 2 * (y_hat - y) / y.size


class BCE(shells.loss):
    def run(self, y, y_hat):
        return (-np.sum(y*np.log(abs(y_hat))+(1-y)*np.log(abs(1-y_hat))))/len(y)

    def derv(self, y, y_hat):
        return (-np.sum(y*np.log(abs(y_hat))+(1-y)*np.log(abs(1-y_hat))))/len(y)
