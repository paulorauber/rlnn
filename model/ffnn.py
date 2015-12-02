import numpy as np

from sklearn.utils import check_random_state
from scipy.special import expit


def sigmoid_prime(z):
    p = expit(z)
    return p*(1 - p)

sigmoid = expit


class FeedForwardNetwork:
    """Feedforward neural network with arbitrary number of layers."""
    def __init__(self, n_neurons, learning_rate=3.0, lmbda=5.0,
                 random_state=None):
        self.n_neurons = n_neurons

        self.learning_rate = learning_rate
        self.lmbda = lmbda

        self.random_state = check_random_state(random_state)

        self.init_parameters()

    def init_parameters(self):
        self.n_layers = len(self.n_neurons)

        self.W_ = []
        self.b_ = []
        for i in range(1, self.n_layers):
            sdev = 1.0 / np.sqrt(self.n_neurons[i - 1])
            dim = (self.n_neurons[i], self.n_neurons[i - 1])

            self.W_.append(self.random_state.normal(0, sdev, size=dim))
            self.b_.append(self.random_state.randn(self.n_neurons[i]))

    def forward_pass(self, x):
        z, a = [], [x]

        L = self.n_layers - 2
        for l in range(L):
            z.append(self.W_[l].dot(a[-1]) + self.b_[l])
            a.append(sigmoid(z[-1]))

        z.append(self.W_[L].dot(a[-1]) + self.b_[L])
        a.append(np.array(z[-1]))

        return z, a

    def predict(self, x):
        return self.forward_pass(x)[1][-1]

    def fit_batch(self, X, Y, mask):
        # Computing current partial derivatives of cost wrt parameters
        partial_b = [np.zeros(b.shape, dtype=float) for b in self.b_]
        partial_W = [np.zeros(W.shape, dtype=float) for W in self.W_]

        for i in range(X.shape[0]):
            xi, yi, mi = X[i], Y[i], mask[i]

            z, a = self.forward_pass(xi)
            err = [(a[-1] - yi)*mi]

            for l in range(2, self.n_layers):
                e = self.W_[-l + 1].T.dot(err[0]) * sigmoid_prime(z[-l])
                err.insert(0, e)

            for l in range(0, self.n_layers - 1):
                partial_b[l] += err[l]

                # Note that a[0] = xi
                partial_W[l] += err[l].reshape(-1, 1).dot(a[l].reshape(1, -1))

        for l in range(0, self.n_layers - 1):
            partial_b[l] /= X.shape[0]
            partial_W[l] /= X.shape[0]

        # Moving parameters in opposite direction to the gradient
        self.move_parameters(partial_b, partial_W)

    def move_parameters(self, partial_b, partial_W):
        decay = (1. - (self.learning_rate * self.lmbda))

        for l in range(0, self.n_layers - 1):
            self.b_[l] -= self.learning_rate * partial_b[l]
            self.W_[l] = decay * self.W_[l] - self.learning_rate*partial_W[l]


class FeedForwardNetworkMomentum(FeedForwardNetwork):
    """Feedforward artificial neural network with arbitrary number of layers.
    Momentum-based stochastic gradient descent."""
    def __init__(self, n_neurons, learning_rate=3.0, lmbda=5.0, mu=0.5,
                 random_state=None):
        FeedForwardNetwork.__init__(self, n_neurons, learning_rate,
                                    lmbda, random_state)
        self.mu = mu

    def init_parameters(self):
        self.n_layers = len(self.n_neurons)

        self.W_ = []
        self.b_ = []

        self.Vw_ = []
        self.Vb_ = []

        for i in range(1, self.n_layers):
            sdev = 1.0 / np.sqrt(self.n_neurons[i - 1])
            dim = (self.n_neurons[i], self.n_neurons[i - 1])

            self.W_.append(self.random_state.normal(0, sdev, size=dim))
            self.b_.append(self.random_state.randn(self.n_neurons[i]))

            self.Vw_.append(np.zeros(dim))
            self.Vb_.append(np.zeros(self.n_neurons[i]))

    def move_parameters(self, partial_b, partial_W):
        for l in range(0, self.n_layers - 1):
            self.Vb_[l] = self.mu*self.Vb_[l] - self.learning_rate*partial_b[l]
            self.Vw_[l] = self.mu*self.Vw_[l] -\
                self.learning_rate*(partial_W[l] + self.lmbda*self.W_[l])

            self.b_[l] += self.Vb_[l]
            self.W_[l] += self.Vw_[l]
