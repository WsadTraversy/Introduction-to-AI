import numpy as np

class MLP:
    def __init__(self, input, output, hid_layers, neurons, activation="ReLU", problem_type="classification"):
        self.input = input
        self.output = output
        self.hid_layers = hid_layers
        self.neurons = neurons

        if activation not in ("ReLU", "sigmoid", "tanh"):
            raise ValueError('You have to choose beetwen: "ReLU", "sigmoid", "tanh"')
        self.activation = activation
        
        if problem_type not in ('regression', 'classification'):
            raise ValueError('You have to choose beetwen: "regression" or "classification"')
        self.problem_type = problem_type

        self._z = [None] * (hid_layers+1)
        self._a = [None] * (hid_layers)
        self._dz = [None] * (hid_layers+1)
        self._da = [None] * (hid_layers)
        self._dw = [None] * (hid_layers+1)
        self._db = [None] * (hid_layers+1)

        self._loss = []
        self._initialized = False


    def _initialize_weights(self):
        weights_holder = [np.random.normal(loc=0.5, scale=0.5, size=(self.neurons, self.input))]
        bias_holder = [np.random.normal(loc=0.5, scale=0.5, size=(self.neurons, 1))]
        for _ in range(self.hid_layers-1):
            weights_holder.append(np.random.normal(loc=0.5, scale=0.5, size=(self.neurons, self.neurons)))
            bias_holder.append(np.random.normal(loc=0.5, scale=0.5, size=(self.neurons, 1)))
        weights_holder.append(np.random.normal(loc=0.5, scale=0.5, size=(self.output, self.neurons)))
        bias_holder.append(np.random.normal(loc=0.5, scale=0.5, size=(self.output, 1)))
            
        self._weights = weights_holder.copy()
        self._bias = bias_holder.copy() 


    def loss(self):
        loss_values = np.array(self._loss)
        self._loss = []
        return loss_values


    def _cost_function(self, h, y):
        if self.problem_type == "regression":
            return (1/2)*(np.sum(h - y.T)**2)
        if self.problem_type == "classification":
            return  -np.sum(y.T*np.log(h)+(1-y.T)*np.log(1-h))


    def _activation_function(self, z):
        if self.activation == "ReLU":
            return np.maximum(0, z)
        elif self.activation == "sigmoid":
            return 1/(1+np.exp(-z))
        elif self.activation == "tanh":
            return np.tanh(z)
    

    def _activation_derivative(self, z):
        holder = []
        if self.activation == "ReLU":
            for item in z:
                holder.append(1) if item>0 else holder.append(0)
        elif self.activation == "sigmoid":
            for item in z:
                sigma = self._activation_function(item)
                holder.append(sigma*(1-sigma))
        elif self.activation == "tanh":
            for item in z:
                holder.append(1-self._activation_function(item)**2)
        return np.array(holder).reshape(-1, 1)


    def _forward(self, X):
        self._z[0] = ((self._weights[0]@X.T+self._bias[0]).reshape(-1, 1))
        self._a[0] = (self._activation_function(self._z[0]))
        
        for i in range(1, len(self._weights)-1):
            self._z[i] = (self._weights[i]@self._a[i-1]+self._bias[i])
            self._a[i] = (self._activation_function(self._z[i]))
        
        self._z[-1] = (self._weights[-1]@self._a[-1]+self._bias[-1])
        if self.problem_type == "classification":
            self._z[-1] = 1/(1+np.exp(-self._z[-1]))


    def _backward(self, X, y):
        self._dz[-1] = (self._z[-1]-y.T)
        self._db[-1] = self._dz[-1]
        self._da[-1] = (self._weights[-1].T@self._dz[-1])

        i=len(self._weights)-2
        while i >= 0:
            self._da[i] = self._weights[i+1].T@self._dz[i+1]
            self._dz[i] = self._da[i]*self._activation_derivative(self._z[i])
            self._dw[i+1] = (self._dz[i+1]@self._a[i].T)
            self._db[i] = self._dz[i]
            i-=1
        self._dw[0] = self._dz[0]@X

        
    def training(self, X, y, learning_rate=0.01):
        X = np.array(X)
        y = np.array(y)

        if not self._initialized:
            self._initialize_weights()
            self._initialized = True

        self._forward(X)
        loss = self._cost_function(self._z[-1], y)
        self._loss.append(loss)
        self._backward(X, y)
        for i, layer in enumerate(self._weights):
            layer -= learning_rate*self._dw[i]
            self._bias[i] -= learning_rate*self._db[i]

        self._z = [None] * (self.hid_layers+1)
        self._a = [None] * (self.hid_layers)
        self._dz = [None] * (self.hid_layers+1)
        self._da = [None] * (self.hid_layers)
        self._dw = [None] * (self.hid_layers+1)
        self._db = [None] * (self.hid_layers+1)


    def evaluate(self, X):
        X = np.array(X)
        
        X = X if len(X.shape) > 1 and X.shape[1] > 0 else np.expand_dims(X, axis=0)

        self._forward(X)
        return self._z[-1]
