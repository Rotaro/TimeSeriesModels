import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Embedding, RNN
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import RandomUniform, constant
from tensorflow.keras.optimizers import Adam


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


class ESRNN(Layer):
    def __init__(self, trend="additive", season="multiplicative", seasonal_period=12, **kwargs):
        assert trend in (None, "additive", "multiplicative"), "Invalid trend (%s)!" % trend
        assert season in (None, "additive", "multiplicative"), "Invalid season (%s)!" % season
        assert season is None or seasonal_period is not None, "Invalid seasonal period (%s)!" % seasonal_period

        self.trend = trend
        self.season = season
        self.seasonal_period = seasonal_period

        super(ESRNN, self).__init__(**kwargs)

        self.state_size = (1, 1, seasonal_period)  # l0, b0, s0
        self.output_size = 1

    def build(self, input_shape):
        # Initial values copied from statsmodels implementation
        init_alpha = 0.5 / max(self.state_size[-1], 1)
        # Smoothing
        self.alpha = self.add_weight(shape=(input_shape[-1], 1),
                                     initializer=constant(init_alpha), name='alpha',
                                     constraint=AbsoluteMinMax(0.0, 1.0))
        # Trend
        self.beta = self.add_weight(shape=(input_shape[-1], 1),
                                    initializer=constant(0.1 * init_alpha), name='beta',
                                    constraint=AbsoluteMinMax(0.0, 1.0))
        # Trend damping
        self.phi = self.add_weight(shape=(input_shape[-1], 1), initializer=constant(0.99), name='phi',
                                   constraint=AbsoluteMinMax(0.8, 1.0))

        # Seasonality smoothing
        self.gamma = self.add_weight(shape=(input_shape[-1], 1), initializer=constant(0.05 * init_alpha), name='gamma',
                                     constraint=AbsoluteMinMax(0.0, 1.0))

        self.built = True

    def call(self, inputs, states):
        inputs = inputs[0]

        if self.season is not None:
            l0, b0, s0 = states[0], states[1], states[2]

            if self.season == "multiplicative" and self.trend == "additive":
                l = self.alpha * inputs / s0[:, :1] + (1 - self.alpha) * (l0 + self.phi * b0)
                b = self.beta * (l - l0) + (1 - self.beta) * b0 * self.phi
                c = self.gamma * inputs / l + (1 - self.gamma) * s0[:, :1]

            # Recreate seasonality states for next iteration so that first element matches next step
            s = tf.keras.layers.concatenate([s0[:, 1:], c], axis=1)

            return (l + b * self.phi) * s[:, :1], [l, b, s]


class ExponentialSmoothing:
    """Exponential smoothing using keras RNN layers.

     Using keras layers in anticipation of sharing information between time series.
     """

    def __init__(self, trend="additive", season="multiplicative", seasonal_period=12):
        assert trend in (None, "additive", "multiplicative"), "Invalid trend (%s)!" % trend
        assert season in (None, "additive", "multiplicative"), "Invalid season (%s)!" % season
        assert season is None or seasonal_period is not None, "Invalid seasonal period (%s)!" % seasonal_period

        assert trend == "additive" and season == "multiplicative", \
            "Only additive trend and multiplicative season supported at the moment."

        self.trend = trend
        self.season = season
        self.seasonal_period = seasonal_period

        self.y = None      # Need to save y for predicting out-of-sample
        self.rnnmodel = None

        self.model = None
        self.predictor = None

    def fit(self, y, lr=1e-2, epochs=100, verbose=2):
        self.y = y.copy()[None, :, None]

        self.model, self.predictor = self._create_model_predictor(lr)
        self.model.fit([
            self.y[:, :-1, :],   # Lagged sales as input
            np.array([0])        # Dummy ID
        ], self.y, epochs=epochs, verbose=verbose)

    def predict(self, n_oos_steps):
        preds, l, b, s = self.predictor.predict([self.y, np.array([0])])
        _, _, _, alpha, beta, phi, gamma = map(lambda x: x.ravel(), self.predictor.get_weights())
        preds_oos = (l + np.arange(1, n_oos_steps) * b * phi ** np.arange(1, n_oos_steps)).ravel() \
                     * np.tile(np.roll(s, -1).ravel(), max(int(n_oos_steps // s.size), 1))[:n_oos_steps - 1]

        return np.hstack([preds.ravel(), preds_oos.ravel()])

    def get_params(self):
        param_names = ["l0", "b0", "s0", "alpha", "beta", "phi", "gamma"]
        return dict(zip(param_names, map(lambda x: x.ravel(), self.predictor.get_weights())))

    def _create_model_predictor(self, lr):
        # Initial values from statsmodels implementation
        l0_start = self.y.ravel()[np.arange(self.y.size) % self.seasonal_period == 0].mean()
        b0_start = self.y.ravel()[1] - self.y.ravel()[0]
        s0_start = self.y.ravel()[:self.seasonal_period] / l0_start

        inp_y = Input(shape=(None, 1))
        inp_emb_id = Input(shape=(1,))  # Dummy ID for embeddings

        # (Ab)using embeddings here for initial value variables
        init_l0 = Embedding(1, 1, embeddings_initializer=constant(l0_start))(inp_emb_id)[:, 0, :]
        init_b0 = Embedding(1, 1, embeddings_initializer=constant(b0_start))(inp_emb_id)[:, 0, :]
        init_seas_emb = Embedding(1, self.seasonal_period, embeddings_initializer=RandomUniform(0.8, 1.2))
        init_seas = init_seas_emb(inp_emb_id)[:, 0, :]

        rnncell = ESRNN(self.trend, self.season, self.seasonal_period)
        rnn = RNN(rnncell, return_sequences=True, return_state=True)
        out_rnn = rnn(inp_y, initial_state=[init_l0, init_b0, init_seas])

        out = tf.keras.layers.concatenate([
            tf.math.reduce_sum((init_l0 + init_b0) * init_seas[:, :1], axis=1)[:, None, None],
            out_rnn[0]
        ], axis=1)

        model = Model(inputs=[inp_y, inp_emb_id], outputs=out)
        model.compile(Adam(lr), "mse")

        # predictor also outputs final state for predicting out-of-sample
        predictor = Model(inputs=[inp_y, inp_emb_id], outputs=[out, out_rnn[1:]])

        # Assign initial seasonality weights
        init_seas_emb.set_weights([s0_start.reshape(init_seas_emb.get_weights()[0].shape)])

        return model, predictor
