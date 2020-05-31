import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

TRAIN_ITERS = 1750
RETRAIN_ITERS = 30
ETA = 0.02
NEURONS_LI = 1
NEURONS_LO = 1
NEURONS_T2_L2 = 5
NEURONS_T3_L2 = 10
NEURONS_T3_L3 = 10

# -------------------
# Sigmoid, RELU and Tanh functions

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1.0 - x)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, 1.0, 0.0)

tanh = np.tanh

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1.0 - x ** 2

# -------------------
# Layer & NeuralNetwork Class

class Layer:
    def __init__(self, data, weights):
        self.weights: np.ndarray = weights
        self.vals: np.ndarray = np.zeros(shape=(data[1]))

        if data[0] == "sigmoid":
            self.func, self.func_der = sigmoid, sigmoid_derivative
        elif data[0] == "relu":
            self.func, self.func_der = relu, relu_derivative
        elif data[0] == "tanh":
            self.func, self.func_der = tanh, tanh_derivative

class NeuralNetwork:
    def __init__(self, layers: dict):
        self.output = np.zeros((NEURONS_LO, 1))
        self.create([layers[layer] for layer in layers])
        self.eta = ETA

    def feedforward(self, X):
        # Setting first Layer to given matrix 'X'
        self.layers[0].vals = X

        for layer, layer_next in zip(self.layers[0:-1], self.layers[1:]):
            layer_next.vals = layer.func(np.dot(layer.vals, layer.weights.T))

        # Last Layer values and weights 
        self.output = self.layers[-1].func(np.dot(self.layers[-1].vals, self.layers[-1].weights.T))

    def backprop(self, y):
        deltas = []
        
        # Last Layer diff
        delta = (y - self.output) * self.layers[-1].func_der(self.output)
        deltas.append(self.eta * np.dot(delta.T, self.layers[-1].vals))

        # Going backwards from last Layer (notice: reversed arrays)
        for layer, layer_prev in zip(reversed(self.layers[0:-1]), reversed(self.layers[1:])):
            delta = layer.func_der(layer_prev.vals) * np.dot(delta, layer_prev.weights)
            deltas.append(self.eta * np.dot(delta.T, layer.vals))

        # Updating weights
        for layer, d_weights in zip(self.layers, reversed(deltas)):
            layer.weights += d_weights

    def create(self, t_layers):
        self.layers = []

        # Connecting Layers with each other
        for layer, layer_next in zip(t_layers[0:-1], t_layers[1:]):
            self.layers.append(Layer(layer,
                np.random.standard_normal((layer_next[1], layer[1]))))

        # Connecting last Layer to output vertex 'y'
        self.layers.append(Layer(t_layers[-1],
            np.random.standard_normal((self.output.shape[1], t_layers[-1][1]))))

    def train(self, X, y):
        for _ in range(TRAIN_ITERS):
            self.feedforward(X)
            self.backprop(y)

    def predict(self, X):
        self.feedforward(X)

# -------------------
# Function to Test out different combinations of NeuralNetwork funcs

def learn_and_test(func: dict, func_name: str, diagnose={}):
    x_sc, y_sc = MinMaxScaler((0, 1)), MinMaxScaler((0, 1))
    x_test, y_test = func["Xpred"].reshape((-1, 1)), func["yansw"].reshape((-1, 1))
    x_train = x_sc.fit_transform(func["X"].reshape((-1, 1)))
    y_train = y_sc.fit_transform(func["y"].reshape((-1, 1)))

    nn = NeuralNetwork(func["layers"])
    plt.scatter(x_test, y_test, 3)

    for idx in range(1, RETRAIN_ITERS+1):
        nn.train(x_train, y_train)
        nn.predict(x_sc.fit_transform(x_test))
        draw_graph(x_test, y_test, y_sc, idx, nn, func_name)
        plt.pause(0.01)
    plt.show()

    plt.scatter(x_test, y_test, 3)
    draw_graph(x_test, y_test, y_sc, RETRAIN_ITERS, nn, func_name)
    plt.show()

# -------------------
# Animate

def draw_graph(x_test, y_test, y_sc, idx, nn, func_name):
    plt.scatter(x_test, y_sc.inverse_transform(nn.output), 3)
    error = "{:.5f}".format(np.square(y_test - y_sc.inverse_transform(nn.output)).mean())
    trained = f"Trained Times: ({TRAIN_ITERS*idx} / {RETRAIN_ITERS*TRAIN_ITERS})"
    error = f"Cumulative Error: {error}"
    plt.title(f"{func_name}\n{trained}\n{error}")

# -------------------
# Main

if __name__ == "__main__":
    
    # Dict {Parabole and Sin Functions} w/ training matrix and label vector + shuffled test cases
    funcs = {

        # Task #2 -> 1 | 5 | 1
        "Parabole Task #2" : {
            "X"     : np.linspace(-50, 50, 26),
            "y"     : np.linspace(-50, 50, 26) ** 2,
            "Xpred" : np.linspace(-50, 50, 101),
            "yansw" : np.linspace(-50, 50, 101) ** 2,
            "layers": {
                "layer1" : ['sigmoid', NEURONS_LI],
                "layer2" : ['sigmoid', NEURONS_T2_L2]
            }},
        
        "Sin Task #2" : {
            "X"     : np.linspace(0, 2, 21),
            "y"     : np.sin((3 * np.pi / 2) * np.linspace(0, 2, 21)),
            "Xpred" : np.linspace(0, 2, 161),
            "yansw" : np.sin((3 * np.pi / 2) * np.linspace(0, 2, 161)),
            "layers": {
                "layer1" : ['tanh', NEURONS_LI],
                "layer2" : ['tanh', NEURONS_T2_L2]
            }},

        # Task #3 -> 1 | 10 | 10 | 1
        "Parabole Task #3" : {
            "X"     : np.linspace(-50, 50, 26),
            "y"     : np.linspace(-50, 50, 26) ** 2,
            "Xpred" : np.linspace(-50, 50, 101),
            "yansw" : np.linspace(-50, 50, 101) ** 2,
            "layers": {
                "layer1" : ['sigmoid', NEURONS_LI],
                "layer2" : ['sigmoid', NEURONS_T3_L2],
                "layer3" : ['sigmoid', NEURONS_T3_L3]
            }},
        
        "Sin Task #3" : {
            "X"     : np.linspace(0, 2, 21),
            "y"     : np.sin((3 * np.pi / 2) * np.linspace(0, 2, 21)),
            "Xpred" : np.linspace(0, 2, 161),
            "yansw" : np.sin((3 * np.pi / 2) * np.linspace(0, 2, 161)),
            "layers": {
                "layer1" : ['tanh', NEURONS_LI],
                "layer2" : ['tanh', NEURONS_T3_L2],
                "layer3" : ['tanh', NEURONS_T3_L3]
            }}
    }

    for func in funcs:
        learn_and_test(funcs[func], func)

# -------------------
# Comment about the results
#
# - Well, some of the functions are providing terrible results.
# - Adding another hidden layer improves predictions. 