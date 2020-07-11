import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Embedding, RNN
from tensorflow.keras.initializers import constant
from tensorflow.keras.optimizers import Adam


from models.tf_model import AbsoluteMinMax, TFTimeSeriesModel


class CrostonsRNN(Layer):
    def __init__(self, sba=False, **kwargs):
        super(CrostonsRNN, self).__init__(**kwargs)

        self.sba = sba
        self.state_size = (1, 1, 1)  # Intensity, delay and actual delay
        self.output_size = 1

    def build(self, input_shape):
        # Typical value from literature
        init_alpha = 0.1
        # Smoothing
        self.alpha = self.add_weight(shape=(input_shape[-1], 1),
                                     initializer=constant(init_alpha), name='alpha',
                                     constraint=AbsoluteMinMax(0.0, 1.0))
        self.built = True

    def call(self, inputs, states):
        # See https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf for details.
        X = inputs[0]

        Z = states[0]
        V = states[1]
        q = states[2]

        Z_next = self.alpha * X + (1 - self.alpha) * Z
        V_next = self.alpha * q + (1 - self.alpha) * V

        zeros_bool = tf.math.equal(inputs, 0)
        zeros = tf.cast(zeros_bool, tf.float32)
        non_zeros = tf.cast(~zeros_bool, tf.float32)

        Z_next = Z_next * non_zeros + Z * zeros
        V_next = V_next * non_zeros + V * zeros
        q = 1 * non_zeros + (q + 1) * zeros

        if self.sba:
            out = (1 - self.alpha / 2) * Z_next / V_next
        else:
            out = Z_next / V_next

        return out, [Z_next, V_next, q]


class CrostonsMethod(TFTimeSeriesModel):
    """Croston's method using keras RNN layers.

     Using keras layers in anticipation of sharing information between time series.

     See https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf for model details.

     :param sba: bool, whether to use SBA (Syntetos-Boylan Approximation) adjustment.
     :param loss: string, loss to use when training model, e.g. mse or poisson
     """

    param_names = ["alpha", "l0"]

    def __init__(self, sba=False, loss="mse"):
        self.y = None         # Need to save y for predicting out-of-sample
        self.rnnmodel = None

        self.sba = sba
        self.loss = loss

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
        preds, *[Z, V, q] = self.predictor.predict([self.y, np.array([0])])

        return np.hstack([preds.ravel(), np.repeat(preds.ravel()[-1], n_oos_steps - 1)])

    def _create_model_predictor(self, lr):
        y = self.y.ravel()
        l0_start = y[0]

        inp_y = Input(shape=(None, 1))
        inp_emb_id = Input(shape=(1,))  # Dummy ID for embeddings

        # (Ab)using embeddings here for initial value variables
        init_l0 = Embedding(1, 1, embeddings_initializer=constant(l0_start), name="l0")(inp_emb_id)[:, 0, :]

        rnncell = CrostonsRNN(self.sba)
        rnn = RNN(rnncell, return_sequences=True, return_state=True)
        out_rnn = rnn(inp_y, initial_state=[init_l0, tf.ones_like(inp_emb_id), tf.ones_like(inp_emb_id)])

        out = tf.keras.layers.concatenate([
            init_l0[:, :, None],
            out_rnn[0]
        ], axis=1)

        model = Model(inputs=[inp_y, inp_emb_id], outputs=out)
        model.compile(Adam(lr), self.loss)

        # predictor also outputs final state for predicting out-of-sample
        predictor = Model(inputs=[inp_y, inp_emb_id], outputs=[out, out_rnn[1:]])

        return model, predictor


class TSBRNN(Layer):
    def __init__(self, **kwargs):
        super(TSBRNN, self).__init__(**kwargs)

        self.state_size = (1, 1)  # Intensity, delay and actual delay
        self.output_size = 1

    def build(self, input_shape):
        # Typical value from literature
        init_alpha, init_beta = 0.1, 0.1

        # Smoothing for sales
        self.alpha = self.add_weight(shape=(input_shape[-1], 1),
                                     initializer=constant(init_alpha), name='alpha',
                                     constraint=AbsoluteMinMax(0.0, 1.0))
        # Smoothing for intervals
        self.beta = self.add_weight(shape=(input_shape[-1], 1),
                                    initializer=constant(init_beta), name='beta',
                                    constraint=AbsoluteMinMax(0.0, 1.0))
        self.built = True

    def call(self, inputs, states):
        # See https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf for details.
        X = inputs[0]

        Z = states[0]
        P = states[1]

        zeros_bool = tf.math.equal(inputs, 0)
        zeros = tf.cast(zeros_bool, tf.float32)
        non_zeros = tf.cast(~zeros_bool, tf.float32)

        Z_next = self.alpha * X + (1 - self.alpha) * Z
        Z_next = Z_next * non_zeros + Z * zeros

        P_next = non_zeros * self.beta + (1 - self.beta) * P

        return P_next * Z_next, [P_next, Z_next]


class TSB(TFTimeSeriesModel):
    """Teunter,Syntetos and Babai (TSB) method using keras RNN layers.

    Using keras layers in anticipation of sharing information between time series.

    See https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf for model details.
    """

    param_names = ["alpha", "beta", "l0"]

    def __init__(self, loss="mse"):
        self.y = None         # Need to save y for predicting out-of-sample
        self.rnnmodel = None

        self.loss = loss

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
        preds, *[Z, V] = self.predictor.predict([self.y, np.array([0])])

        return np.hstack([preds.ravel(), np.repeat(preds.ravel()[-1], n_oos_steps - 1)])

    def _create_model_predictor(self, lr):
        y = self.y.ravel()
        l0_start = y[0]

        inp_y = Input(shape=(None, 1))
        inp_emb_id = Input(shape=(1,))  # Dummy ID for embeddings

        # (Ab)using embeddings here for initial value variables
        init_l0 = Embedding(1, 1, embeddings_initializer=constant(l0_start), name="l0")(inp_emb_id)[:, 0, :]

        rnn = RNN(TSBRNN(), return_sequences=True, return_state=True)
        out_rnn = rnn(inp_y, initial_state=[init_l0, tf.ones_like(inp_emb_id)])

        out = tf.keras.layers.concatenate([
            init_l0[:, :, None],
            out_rnn[0]
        ], axis=1)

        model = Model(inputs=[inp_y, inp_emb_id], outputs=out)
        model.compile(Adam(lr), self.loss)

        # predictor also outputs final state for predicting out-of-sample
        predictor = Model(inputs=[inp_y, inp_emb_id], outputs=[out, out_rnn[1:]])

        return model, predictor
