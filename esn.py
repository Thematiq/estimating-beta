from auto_esn.esn.esn import DeepESN
from auto_esn.esn.reservoir import activation as A
from auto_esn.esn.reservoir.initialization import CompositeInitializer, WeightInitializer
import torch
import numpy as np 
import pandas as pd

device = torch.device('cpu')
dtype = torch.double
torch.set_default_dtype(dtype)
class ESN(DeepESN):
    def __init__(self, input_size: int = 1, hidden_size: int = 500, output_dim: int = 1, bias: bool = False, initializer: WeightInitializer = WeightInitializer(), num_layers=2, activation=A.self_normalizing_default(), washout: int = 30, regularization: float = 1):
        super().__init__(input_size, hidden_size, output_dim, bias, initializer, num_layers, activation, washout, regularization)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.bias = bias
        self.initializer = initializer
        self.num_layers= num_layers
        self.activation = activation
        self.washout = washout
        self.regularization = regularization

    def fit(self, X, y): 
        if  isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            y = y.to_numpy()
        return super().fit(torch.from_numpy(X), torch.from_numpy(y))

    def predict(self, X):
        if  isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        mapped_input = self.reservoir(torch.from_numpy(X))
        return self.readout(mapped_input)[:,-X.shape[0]]
        # return np.squeeze(self(torch.tensor(X, device=device)))
    
    def get_params(self, deep=False):
        return {"input_size": self.input_size, "hidden_size": self.hidden_size,"output_dim": self.output_dim, "bias": self.bias,"initializer": self.initializer, "num_layers": self.num_layers,"activation": self.activation, "washout": self.washout, "regularization": self.regularization}
    
    def set_params(self, **params):
        return ESN(**params)