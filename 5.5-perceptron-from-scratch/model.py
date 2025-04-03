import numpy as np

class Perceptron:
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

        self._weights = []
        self._bias = []

        self._z = []
        self._a = []
        self._dz = []
        self._da = []
        self._dw = []
        self._db = []

        self._loss = []
        self._initialized = False


    def _initialize_weights(self, batch_size):
            weights_holder = [np.random.normal(loc=0.5, scale=0.5, size=(self.neurons, self.input))]
            bias_holder = [np.random.normal(loc=0.5, scale=0.5, size=(self.neurons, 1))]
            for _ in range(self.hid_layers-1):
                weights_holder.append(np.random.normal(loc=0.5, scale=0.5, size=(self.neurons, self.neurons)))
                bias_holder.append(np.random.normal(loc=0.5, scale=0.5, size=(self.neurons, 1)))
            weights_holder.append(np.random.normal(loc=0.5, scale=0.5, size=(self.output, self.neurons)))
            bias_holder.append(np.random.normal(loc=0.5, scale=0.5, size=(self.output, 1)))
            
            self._weights = [weights_holder.copy() for _ in range(batch_size)]
            self._bias = [bias_holder.copy() for _ in range(batch_size)]


    def loss(self):
        loss_values = np.array(self._loss)
        self._loss = []
        return loss_values


    def _cost_function(self, h, y):
        if self.problem_type == "regression":
            return (1/2)*(np.sum(h - y)**2)
        if self.problem_type == "classification":
            epsilon = 1e-10 
            h = np.clip(h, epsilon, 1 - epsilon)  # Ensure 0 < h < 1
            return  -np.sum(y*np.log(h)+(1-y)*np.log(1-h))


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
        return np.array(holder)


    def _forward(self, X):
        for j, el in enumerate(X):
            z_holder = [None] * (self.hid_layers+1)
            a_holder = [None] * (self.hid_layers)

            current_weights = self._weights[j]
            current_bias = self._bias[j]

            z_holder[0] = ((current_weights[0]@el.T+current_bias[0].T).reshape(-1, 1))
            a_holder[0] = (self._activation_function(z_holder[0]))
            
            for i in range(1, len(current_weights)-1):
                z_holder[i] = (current_weights[i]@a_holder[i-1]+current_bias[i])
                a_holder[i] = (self._activation_function(z_holder[i]))
    
            z_holder[-1] = (current_weights[-1]@a_holder[-1]+current_bias[-1])
            if self.problem_type == "classification":
                z_holder[-1] = 1/(1+np.exp(-z_holder[-1]))
            
            self._z.append(z_holder)
            self._a.append(a_holder)


    def _backward(self, X, y):
        for j, el in enumerate(X):
            dz_holder = [None] * (self.hid_layers+1)
            da_holder = [None] * (self.hid_layers)
            dw_holder = [None] * (self.hid_layers+1)
            db_holder = [None] * (self.hid_layers+1)

            current_weights = self._weights[j]
            if y.shape[1] == 1:
                dz_holder[-1] = (self._z[j][-1]-y)
            else:
                y_transformed = y[j].reshape(-1, 1)
                dz_holder[-1] = (self._z[j][-1]-y_transformed)
            db_holder[-1] = dz_holder[-1]
            da_holder[-1] = (current_weights[-1].T@dz_holder[-1])

            i=len(current_weights)-2
            while i >= 0:
                da_holder[i] = current_weights[i+1].T@dz_holder[i+1]
                dz_holder[i] = da_holder[i]*self._activation_derivative(self._z[j][i]).reshape(-1, 1)
                dw_holder[i+1] = (dz_holder[i+1]@self._a[j][i].T)
                db_holder[i] = dz_holder[i]
                i-=1
            if X.shape[0] == 1:
                dw_holder[0] = dz_holder[0]@X
            else:
                dw_holder[0] = dz_holder[0]@(el.reshape(1, el.shape[0]))

            self._dz.append(dz_holder)
            self._da.append(da_holder)
            self._dw.append(dw_holder)
            self._db.append(db_holder)

        
    def training(self, X, y, learning_rate=0.0001, L2=0):
        X = np.array(X)
        y = np.array(y)
        X = X if len(X.shape) > 1 and X.shape[1] > 0 else np.expand_dims(X, axis=0)
        y = y if len(y.shape) > 1 and y.shape[1] > 0 else y.reshape(-1, 1)
        batch_size = X.shape[0]

        if not self._initialized:
            self._initialize_weights(batch_size=batch_size)
            self._initialized = True

        self._forward(X)
        loss = []
        for i, el in enumerate(self._z):
            if y.shape[1] == 1:
                loss.append(self._cost_function(el[-1], y))
            else:
                loss.append(self._cost_function(el[-1], y[i]))
        self._loss.append(np.array(loss).mean())
        self._backward(X, y)

        dw_mean = [np.zeros((self.neurons, self.input))]
        db_mean = [np.zeros((self.neurons, 1))]
        for _ in range(self.hid_layers-1):
            dw_mean.append(np.zeros((self.neurons, self.neurons)))
            db_mean.append(np.zeros((self.neurons, 1)))
        dw_mean.append(np.zeros((self.output, self.neurons)))
        db_mean.append(np.zeros((self.output, 1)))
        for el in self._weights:
            for i, layer in enumerate(el):
                dw_mean[i] += layer
        for el in self._bias:
            for i, layer in enumerate(el):
                db_mean[i] += layer
       
        for el in self._weights:
            for j, layer in enumerate(el):
                layer -= (learning_rate*dw_mean[j] + L2*layer)
        for el in self._bias:
            for j, bias in enumerate(el):
                bias -= (learning_rate*db_mean[j] + L2*bias)
        self._z = []
        self._a = []
        self._dz = []
        self._da = []
        self._dw = []
        self._db = []


    def evaluate(self, X):
        X = np.array(X)
        
        X = X if len(X.shape) > 1 and X.shape[1] > 0 else np.expand_dims(X, axis=0)

        self._forward(X)
        output = self._z[-1][-1]
        self._z = []
        self._a = []
        return output
