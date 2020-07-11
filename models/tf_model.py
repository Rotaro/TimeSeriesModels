import numpy as np

from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K
from tensorflow.keras.layers import RNN


class AbsoluteMinMax(Constraint):
    def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis

    def __call__(self, w):
        norms = K.sqrt(K.sum(w, axis=self.axis, keepdims=True))
        desired = (self.rate * K.clip(norms, self.min_value, self.max_value) +
                   (1 - self.rate) * norms)
        w = w * (desired / (K.epsilon() + norms))
        return w


class TFTimeSeriesModel:
    def get_param(self, name):
        try:
            param = self.predictor.get_layer(name).weights[0].numpy()
        except ValueError as e:
            # Parameter is in ESRNN layer
            layer = [layer for layer in self.predictor.layers if isinstance(layer, RNN)][0]._layers[0]
            param = getattr(layer, name, None)
            param = param.numpy() if param else np.array([])

        return param

    def get_params(self):
        return dict(zip(self.param_names, map(lambda x: self.get_param(x).ravel(), self.param_names)))
