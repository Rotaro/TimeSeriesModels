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
        self.seasonal = season
        self.seasonal_period = seasonal_period

        super(ESRNN, self).__init__(**kwargs)

        self.state_size = (1,) + ((1,) if self.trend else ()) + ((seasonal_period,) if self.seasonal else ())
        self.output_size = 1

    def build(self, input_shape):
        # Initial values copied from statsmodels implementation
        init_alpha = 0.5 / max(self.state_size[-1], 1)
        # Smoothing
        self.alpha = self.add_weight(shape=(input_shape[-1], 1),
                                     initializer=constant(init_alpha), name='alpha',
                                     constraint=AbsoluteMinMax(0.0, 1.0))
        if self.trend:
            self.beta = self.add_weight(shape=(input_shape[-1], 1),
                                        initializer=constant(0.1 * init_alpha), name='beta',
                                        constraint=AbsoluteMinMax(0.0, 1.0))
            # Trend damping
            self.phi = self.add_weight(shape=(input_shape[-1], 1), initializer=constant(0.99), name='phi',
                                       constraint=AbsoluteMinMax(0.8, 1.0))

        if self.seasonal:
            self.gamma = self.add_weight(shape=(input_shape[-1], 1), initializer=constant(0.5),
                                         name='gamma', constraint=AbsoluteMinMax(0.0, 1.0))

        self.built = True

    def call(self, inputs, states):
        inputs = inputs[0]

        l0 = states[0]
        if self.trend:
            b0 = states[1]
        if self.seasonal:
            s0 = states[1 + (self.trend is not None)]

        if self.seasonal:
            if self.seasonal == "multiplicative":
                if self.trend == "multiplicative":
                    l = self.alpha * inputs / s0[:, :1] + (1 - self.alpha) * (l0 * self.phi * b0)
                    b = self.beta * (l / l0) + (1 - self.beta) * b0 * self.phi
                    s = self.gamma * inputs / (l * b * self.phi) + (1 - self.gamma) * s0[:, :1]
                    out = (l * b * self.phi) * s0[:, 1:2]
                    out_states = [l, b]
                elif self.trend == "additive":
                    l = self.alpha * inputs / s0[:, :1] + (1 - self.alpha) * (l0 + self.phi * b0)
                    b = self.beta * (l - l0) + (1 - self.beta) * b0 * self.phi
                    s = self.gamma * inputs / (l + b * self.phi) + (1 - self.gamma) * s0[:, :1]
                    out = (l + b * self.phi) * s0[:, 1:2]
                    out_states = [l, b]
                else:
                    l = self.alpha * inputs / s0[:, :1] + (1 - self.alpha) * l0
                    s = self.gamma * inputs / l + (1 - self.gamma) * s0[:, :1]
                    out = l * s0[:, 1:2]
                    out_states = [l]

            elif self.seasonal == "additive":
                if self.trend == "multiplicative":
                    l = self.alpha * (inputs - s0[:, :1]) + (1 - self.alpha) * (l0 * self.phi * b0)
                    b = self.beta * (l / l0) + (1 - self.beta) * b0 * self.phi
                    s = self.gamma * (inputs - l * self.phi * b) + (1 - self.gamma) * s0[:, :1]
                    out = l * b * self.phi + s0[:, 1:2]
                    out_states = [l, b]
                elif self.trend == "additive":
                    l = self.alpha * (inputs - s0[:, :1]) + (1 - self.alpha) * (l0 + self.phi * b0)
                    b = self.beta * (l - l0) + (1 - self.beta) * b0 * self.phi
                    s = self.gamma * (inputs - l - self.phi * b) + (1 - self.gamma) * s0[:, :1]
                    out = l + b * self.phi + s0[:, 1:2]
                    out_states = [l, b]
                else:
                    l = self.alpha * (inputs - s0[:, :1]) + (1 - self.alpha) * l0
                    s = self.gamma * (inputs - l) + (1 - self.gamma) * s0[:, :1]
                    out = l + s0[:, 1:2]
                    out_states = [l]

            # Recreate seasonality states for next iteration so that first element matches next step
            s = tf.keras.layers.concatenate([s0[:, 1:], s], axis=1)

            return out, out_states + [s]

        elif self.trend:
            if self.trend == "multiplicative":
                l = self.alpha * inputs + (1 - self.alpha) * (l0 * self.phi * b0)
                b = self.beta * (l / l0) + (1 - self.beta) * b0 * self.phi
                out = (l * b * self.phi)

            elif self.trend == "additive":
                l = self.alpha * inputs + (1 - self.alpha) * (l0 + self.phi * b0)
                b = self.beta * (l - l0) + (1 - self.beta) * b0 * self.phi
                out = (l + b * self.phi)

            return out, [l, b]
        else:
            l = self.alpha * inputs + (1 - self.alpha) * l0

            return l, [l]


class ExponentialSmoothing:
    """Exponential smoothing using keras RNN layers.

     Using keras layers in anticipation of sharing information between time series.
     """

    def __init__(self, trend="additive", damped=True, seasonal="multiplicative", seasonal_period=12):
        assert trend in (None, "additive", "multiplicative"), "Invalid trend (%s)!" % trend
        assert seasonal in (None, "additive", "multiplicative"), "Invalid season (%s)!" % seasonal
        assert seasonal is None or seasonal_period is not None, "Invalid seasonal period (%s)!" % seasonal_period

        # assert trend == "additive" and season in ("additive", "multiplicative"), \
        #     "Only additive trend and seasonality supported at the moment."

        self.trend = trend
        self.seasonal = seasonal
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
        preds, *l_b_s = self.predictor.predict([self.y, np.array([0])])

        if self.seasonal == "multiplicative":
            if self.trend == "additive":
                l, b, s = l_b_s
                phi = self.get_param("phi").ravel()[:, None]

                preds_oos = (l + np.arange(1, n_oos_steps) * b * phi ** np.arange(1, n_oos_steps)).ravel() \
                             * np.tile(np.roll(s, -1).ravel(), max(int(n_oos_steps // s.size), 1))[:n_oos_steps - 1]
            elif self.trend == "multiplicative":
                l, b, s = l_b_s
                phi = self.get_param("phi").ravel()[:, None]

                preds_oos = (l * b * phi ** np.arange(1, n_oos_steps)).ravel() \
                            * np.tile(np.roll(s, -1).ravel(), max(int(n_oos_steps // s.size), 1))[:n_oos_steps - 1]
            else:
                l, s = l_b_s

                preds_oos = l.ravel() * \
                            np.tile(np.roll(s, -1).ravel(), max(int(n_oos_steps // s.size), 1))[:n_oos_steps - 1]
        elif self.seasonal == "additive":
            if self.trend == "additive":
                l, b, s = l_b_s
                phi = self.get_param("phi").ravel()[:, None]

                preds_oos = (l + np.arange(1, n_oos_steps) * b * phi ** np.arange(1, n_oos_steps)).ravel() \
                            + np.tile(np.roll(s, -1).ravel(), max(int(n_oos_steps // s.size), 1))[:n_oos_steps - 1]
            elif self.trend == "multiplicative":
                l, b, s = l_b_s
                phi = self.get_param("phi").ravel()[:, None]

                preds_oos = (l * b * phi ** np.arange(1, n_oos_steps)).ravel() \
                            + np.tile(np.roll(s, -1).ravel(), max(int(n_oos_steps // s.size), 1))[:n_oos_steps - 1]
            else:
                l, s = l_b_s

                preds_oos = l \
                            + np.tile(np.roll(s, -1).ravel(), max(int(n_oos_steps // s.size), 1))[:n_oos_steps - 1]
        elif self.trend:
            if self.trend == "additive":
                l, b = l_b_s
                phi = self.get_param("phi").ravel()[:, None]

                preds_oos = (l + np.arange(1, n_oos_steps) * b * phi ** np.arange(1, n_oos_steps)).ravel()
            elif self.trend == "multiplicative":
                l, b = l_b_s
                phi = self.get_param("phi").ravel()[:, None]

                preds_oos = (l * b * phi ** np.arange(1, n_oos_steps)).ravel()
        else:
            preds_oos = np.repeat(l_b_s[0].ravel(), n_oos_steps - 1)

        return np.hstack([preds.ravel(), preds_oos.ravel()])

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
        param_names = ["l0", "b0", "s0", "alpha", "beta", "phi", "gamma"]
        return dict(zip(param_names, map(lambda x: self.get_param(x).ravel(), param_names)))

    def _create_model_predictor(self, lr):
        y = self.y.ravel()
        # Initial values from statsmodels implementation
        if self.seasonal:
            l0_start = y[np.arange(y.size) % self.seasonal_period == 0].mean()
            lead, lag = y[self.seasonal_period: 2 * self.seasonal_period], y[:self.seasonal_period]

            if self.trend == "multiplicative":
                b0_start = ((lead - lag) / self.seasonal_period).mean()
            elif self.trend == "additive":
                b0_start = np.exp((np.log(lead.mean()) - np.log(lag.mean())) / self.seasonal_period)

            if self.seasonal == "multiplicative":
                s0_start = y[:self.seasonal_period] / l0_start
            elif self.seasonal == "additive":
                s0_start = y[:self.seasonal_period] - l0_start
        elif self.trend:
            l0_start = y[0]
            if self.trend == "multiplicative":
                b0_start = y[1] / l0_start
            elif self.trend == "additive":
                b0_start = y[1] - l0_start
        else:
            l0_start = y[0]

        inp_y = Input(shape=(None, 1))
        inp_emb_id = Input(shape=(1,))  # Dummy ID for embeddings

        # (Ab)using embeddings here for initial value variables
        init_l0 = [Embedding(1, 1, embeddings_initializer=constant(l0_start), name="l0")(inp_emb_id)[:, 0, :]]

        if self.trend:
            init_b0 = [Embedding(1, 1, embeddings_initializer=constant(b0_start), name="b0")(inp_emb_id)[:, 0, :]]
        else:
            init_b0 = []

        if self.seasonal:
            init_seas_emb = Embedding(1, self.seasonal_period, embeddings_initializer=RandomUniform(0.8, 1.2),
                                      name="s0")
            init_seas = [init_seas_emb(inp_emb_id)[:, 0, :]]
        else:
            init_seas = []

        rnncell = ESRNN(self.trend, self.seasonal, self.seasonal_period)
        rnn = RNN(rnncell, return_sequences=True, return_state=True)
        out_rnn = rnn(inp_y, initial_state=init_l0 + init_b0 + init_seas)

        if self.trend == "multiplicative":
            l0_b0 = init_l0[0] * init_b0[0]
        elif self.trend == "additive":
            l0_b0 = init_l0[0] + init_b0[0]
        else:
            l0_b0 = init_l0[0]

        if self.seasonal == "multiplicative":
            l0_b0_s0 = l0_b0 * init_seas[0][:, :1]
        elif self.seasonal == "additive":
            l0_b0_s0 = l0_b0 + init_seas[0][:, :1]
        else:
            l0_b0_s0 = l0_b0

        out = tf.keras.layers.concatenate([
            tf.math.reduce_sum(l0_b0_s0, axis=1)[:, None, None],
            out_rnn[0]
        ], axis=1)

        model = Model(inputs=[inp_y, inp_emb_id], outputs=out)
        model.compile(Adam(lr), "mse")

        # predictor also outputs final state for predicting out-of-sample
        predictor = Model(inputs=[inp_y, inp_emb_id], outputs=[out, out_rnn[1:]])

        # Assign initial seasonality weights
        if self.seasonal:
            init_seas_emb.set_weights([s0_start.reshape(init_seas_emb.get_weights()[0].shape)])

        return model, predictor
