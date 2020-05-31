import numpy as np

TEST_ITERS = 50000
ETA = 0.2
NEURONS = 4

# -------------------
# Sigmoid and RELU functions

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1.0 - x)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, 1.0, 0.0)

# -------------------
# NeuralNetwork Class

class NeuralNetwork:
    def __init__(self, x, y, func):
        self.input = x
        self.weights1 = np.random.rand(NEURONS, self.input.shape[1])
        self.weights2 = np.random.rand(1, NEURONS)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.eta = ETA

        if func == "sigm":
            self.func1, self.func1_der = sigmoid, sigmoid_derivative
            self.func2, self.func2_der = sigmoid, sigmoid_derivative
        elif func == "relu":
            self.func1, self.func1_der = relu, relu_derivative
            self.func2, self.func2_der = relu, relu_derivative
        elif func == "sigm_relu":
            self.func1, self.func1_der = sigmoid, sigmoid_derivative
            self.func2, self.func2_der = relu, relu_derivative
        elif func == "relu_sigm":
            self.func1, self.func1_der = relu, relu_derivative
            self.func2, self.func2_der = sigmoid, sigmoid_derivative
        elif func == "mash1":
            self.func1, self.func1_der = sigmoid, relu_derivative
            self.func2, self.func2_der = relu, sigmoid_derivative
        elif func == "mash2":
            self.func1, self.func1_der = relu, sigmoid_derivative
            self.func2, self.func2_der = relu, sigmoid_derivative


    def feedforward(self):
        self.layer1 = self.func1(np.dot(self.input, self.weights1.T))
        self.output = self.func2(np.dot(self.layer1, self.weights2.T))

    def backprop(self):
        delta2 = (self.y - self.output) * self.func2_der(self.output)
        d_weights2 = self.eta * np.dot(delta2.T, self.layer1)

        delta1 = self.func1_der(self.layer1) * np.dot(delta2, self.weights2)
        d_weights1 = self.eta * np.dot(delta1.T, self.input)

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self):
        for _ in range(TEST_ITERS):
            self.feedforward()
            self.backprop()

    def predict(self, test):
        self.layer1_p = self.func1(np.dot(self.input, self.weights1.T))
        self.output_p = self.func2(np.dot(self.layer1, self.weights2.T))

# -------------------
# Function to Test out differnt combinations of NeuralNetwork funcs

diagnose = {}

def show_diagnose(diagnose: dict):
    print("-> Examine    Total Error Predictions")
    for func in diagnose:
        print(" - {: <10}".format(func), f"{diagnose[func]['train error']}")

def learn_and_test(logic: dict, logic_name: str, diagnose={}):
    print(f"\n[Logic Gate] Testing: {logic_name.upper()}")

    funcs = ["sigm", "relu", "sigm_relu", "relu_sigm"]

    for func in funcs:
        print("[Func] Using: ", func)
        nn = NeuralNetwork(logic["X"], logic["y"], func)
        
        # Training
        nn.train()
        error   = "{:.8f}".format(np.square(logic["y"] - nn.output).mean())
        answer  = [", ".join(str("{:.8f}".format(x[0])) for x in nn.output)] 
        expcted = [", ".join(str("{:.8f}".format(x[0])) for x in logic["y"])]
        print(f"|T| Error:  {error} \n|T| Answer:   {answer},  \n|T| Expected: {expcted}")

        # Predicting
        nn.predict(logic["Xpred"])
        errorp   = "{:.8f}".format(np.square(logic["y"] - nn.output_p).mean())
        answerp  = [", ".join(str("{:.8f}".format(x[0])) for x in nn.output_p)] 
        expctedp = [", ".join(str("{:.8f}".format(x[0])) for x in logic["yansw"])]
        print(f"|P| Error:  {errorp} \n|P| Answer:   {answerp},  \n|P| Expected: {expctedp}\n")

        # Diagnosing
        if func not in diagnose:
            diagnose[func] = {"train error" : 0,  "predi error" : 0}
        diagnose[func]["train error"] += float(error)
        diagnose[func]["predi error"] += float(error)

    show_diagnose(diagnose)

# -------------------
# Main

if __name__ == "__main__":
    
    # Dict {XOR, AND, OR} w/ training matrix and label vector + shuffled test cases
    logics = {
        "xor" : {
            "X"     : np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
            "y"     : np.array([[0], [1], [1], [0]]),
            "Xpred" : np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]]),
            "yansw" : np.array([[0], [0], [1], [1]])},
        
        "and" : {
            "X"     : np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0]]),
            "y"     : np.array([[1], [0], [0], [0]]),
            "Xpred" : np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1]]),
            "yansw" : np.array([[0], [0], [0], [0]])},
        
        "or" : {
            "X"     : np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
            "y"     : np.array([[0], [1], [1], [1]]),
            "Xpred" : np.array([[0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
            "yansw" : np.array([[1], [1], [1], [1]])}
    }

    for logic in logics:
        learn_and_test(logics[logic], logic)

# -------------------
# Comment about the results (as asked in the task)
#
# I'll just use bullet points for my observed conclusions
# Notice: Scores is total acumulated error.
#
# - - - w/ ETA ~ 0.01:
#     - On very low amount (~20) of test iterations RELU gives best result
#     - On medium amount (~5000) of test iterations SIGM_RELU gives best result (~0.24)
#           with RELU being right after with 0.25
#     - On large amount (~50000) of test iterations SIGM_RELU gives perfect result (~0.0)
#
# - - - w/ ETA ~ 0.5:
#     - On very low amount (~10) of test iterations RELU_SIGM gives best result (~0.5)
#     - On medium amount (~5000) of test iterations SIGM gives best result (~0.001)
#     - On large amount (~50000) about the same as above.
#
# Conclusions: It depends. Every case had his best performing spot.