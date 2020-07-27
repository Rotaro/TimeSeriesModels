import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Embedding, RNN
from tensorflow.keras.initializers import constant
from tensorflow.keras.optimizers import Adam


from models.tf_model import AbsoluteMinMax, TFTimeSeriesModel
import models.seasonality as seasonality


class IntermittentDemandModel(TFTimeSeriesModel):
    def __init__(self):
        self.x = None
        self.y = None
        self.seasonality = None

        self.model = None
        self.predictor = None

    def _handle_y(self, y):
        if y.ndim == 1:
            self.y = y.copy()[None, :, None]
        elif y.ndim == 2:
            self.y = y.copy()[:, :, None]
        else:
            raise Exception("Invalid y shape! (%s)" % str(y.shape))

        return y

    def _handle_x(self, x):
        if self.seasonality is not None:
            assert x is not None, "Need x input for seasonality!"
            x = self.seasonality.datetime_to_array(x)
        self.x = x

    def _create_model_predictor(self, lr):
        raise NotImplementedError

    def _get_dummy_ids(self):
        return np.arange(self.y.shape[0], dtype=np.int32)

    def fit(self, y, x=None, lr=1e-2, epochs=100, verbose=2):
        self._handle_y(y)
        self._handle_x(x)

        self.model, self.predictor = self._create_model_predictor(lr)
        self.model.fit(
            [
                self.y[:, :-1, :],          # Lagged sales as input
                self._get_dummy_ids()       # Dummy IDs
            ]
            + (self.seasonality.get_fit_inputs(self._get_dummy_ids(), self.x) if self.seasonality else []),
            self.y, epochs=epochs, verbose=verbose)

    def predict(self, n_oos_steps):
        preds, *_ = self.predictor.predict(
            [self.y, np.arange(self.y.shape[0])]
            + (self.seasonality.get_predict_inputs(self._get_dummy_ids(), self.x) if self.seasonality else [])
        )

        if self.seasonality:
            oos = self.seasonality.get_oos_predictions(preds[:, -1:, 0], self.x[:, -1:], n_oos_steps)
        else:
            oos = np.repeat(preds[:, -1:, 0], n_oos_steps - 1, axis=1)

        return np.hstack([preds[:, :, 0], oos])


class CrostonsRNN(Layer):
    def __init__(self,  n_time_series, sba=False, **kwargs):
        super(CrostonsRNN, self).__init__(**kwargs)

        self.n_time_series = n_time_series
        self.sba = sba
        self.state_size = (1, 1, 1)  # Intensity, delay and actual delay

    def build(self, input_shape):
        # Typical value from literature
        init_alpha = 0.1
        # Smoothing
        self.alpha = self.add_weight(shape=(self.n_time_series, 1),
                                     initializer=constant(init_alpha), name='alpha',
                                     constraint=AbsoluteMinMax(0.0, 1.0))
        self.built = True

    def call(self, inputs, states):
        # See https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf for details.
        X = inputs[0]
        X_id = inputs[1][:, 0]

        Z = states[0]
        V = states[1]
        q = states[2]

        alpha = tf.clip_by_value(tf.gather(self.alpha, X_id), 0.0, 1.0)

        Z_next = alpha * X + (1 - alpha) * Z
        V_next = alpha * q + (1 - alpha) * V

        zeros_bool = tf.math.equal(X, 0)
        zeros = tf.cast(zeros_bool, tf.float32)
        non_zeros = tf.cast(~zeros_bool, tf.float32)

        Z_next = Z_next * non_zeros + Z * zeros
        V_next = V_next * non_zeros + V * zeros
        q = 1 * non_zeros + (q + 1) * zeros

        if self.sba:
            out = (1 - alpha / 2) * Z_next / V_next
        else:
            out = Z_next / V_next

        return out, [Z_next, V_next, q]


class CrostonsMethod(IntermittentDemandModel):
    """Croston's method using keras RNN layers.

     Using keras layers in anticipation of sharing information between time series.

     See https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf for model details.

     :param sba: bool, whether to use SBA (Syntetos-Boylan Approximation) adjustment.
     :param loss: string, loss to use when training model, e.g. mse or poisson
     :param seasonality_type: string, type of multiplicative seasonality to use for all time series.
     """

    param_names = ["alpha", "Z0", "V0"]

    def __init__(self, sba=False, loss="mse", seasonality=None):
        self.y = None         # Need to save y for predicting out-of-sample
        self.x = None         # Need to save x for predicting out-of-sample
        self.rnnmodel = None

        self.sba = sba
        self.loss = loss

        self.model = None
        self.predictor = None

        self.seasonality = seasonality

        # TODO: Add seasonality parameters
        self.param_names = self.param_names

    def _create_model_predictor(self, lr):
        y = self.y
        Z0_start = y[:, 0]
        V0_start = (np.argmax(np.cumsum(y > 0, axis=1) == 2, axis=1)
                    - np.argmax(np.cumsum(y > 0, axis=1) == 1, axis=1)).reshape((y.shape[0], 1))

        inp_y = Input(shape=(None, 1))
        inp_emb_id = Input(shape=(1,))  # Dummy ID for embeddings

        if self.seasonality:
            inp_y_decoded = self.seasonality.apply_decoder(inp_y)
        else:
            inp_y_decoded = inp_y

        # (Ab)using embeddings here for initial value variables
        init_Z0 = Embedding(y.shape[0], 1, embeddings_initializer=constant(Z0_start), name="Z0")(inp_emb_id)[:, 0, :]
        init_V0 = Embedding(y.shape[0], 1, embeddings_initializer=constant(V0_start), name="V0")(inp_emb_id)[:, 0, :]

        rnncell = CrostonsRNN(y.shape[0], self.sba)
        rnn = RNN(rnncell, return_sequences=True, return_state=True)
        out_rnn = rnn((inp_y_decoded, tf.cast(inp_emb_id[:, None, :] * tf.ones_like(inp_y_decoded), tf.int32)),
                      initial_state=[init_Z0, init_V0, tf.ones_like(inp_emb_id)])

        if self.sba:
            initial_out = ((init_Z0 / init_V0) * (1 - rnncell.alpha / 2))
        else:
            initial_out = (init_Z0 / init_V0)

        out = tf.keras.layers.concatenate([
            initial_out[:, :, None],
            out_rnn[0]
        ], axis=1)

        if self.seasonality:
            out = self.seasonality.apply_encoder(out)

        model = Model(inputs=[inp_y, inp_emb_id] + (self.seasonality.get_model_inputs() if self.seasonality else []),
                      outputs=out)
        model.compile(Adam(lr), self.loss)

        # predictor also outputs final state for predicting out-of-sample
        predictor = Model(
            inputs=[inp_y, inp_emb_id] + (self.seasonality.get_model_inputs() if self.seasonality else []),
            outputs=[out, out_rnn[1:]]
        )

        return model, predictor

    def __repr__(self):
        return "%s - sba: %s (%s)" % (
            self.__class__.__name__, self.sba,
             ",".join(["%s=%s" % (k, np.round(v.ravel() if v.size > 0 else 0, 2)) for k, v in self.get_params().items()])
    )


class TSBRNN(Layer):
    def __init__(self, n_time_series, **kwargs):
        super(TSBRNN, self).__init__(**kwargs)

        self.n_time_series = n_time_series
        self.state_size = (1, 1)  # Intensity, delay

    def build(self, input_shape):
        # Typical value from literature
        init_alpha, init_beta = 0.05, 0.05

        # Smoothing for sales
        self.alpha = self.add_weight(shape=(self.n_time_series, 1),
                                     initializer=constant(init_alpha), name='alpha',
                                     constraint=AbsoluteMinMax(0.0, 1.0))
        # Smoothing for intervals
        self.beta = self.add_weight(shape=(self.n_time_series, 1),
                                    initializer=constant(init_beta), name='beta',
                                    constraint=AbsoluteMinMax(0.0, 1.0))
        self.built = True

    def call(self, inputs, states):
        # See https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf for details.
        X = inputs[0]
        X_id = inputs[1][:, 0]

        Z = states[0]
        P = states[1]

        zeros_bool = tf.math.equal(X, 0)
        zeros = tf.cast(zeros_bool, tf.float32)
        non_zeros = tf.cast(~zeros_bool, tf.float32)

        alpha = tf.gather(self.alpha, X_id)
        beta = tf.gather(self.beta, X_id)

        Z_next = alpha * X + (1 - alpha) * Z
        Z_next = Z_next * non_zeros + Z * zeros

        P_next = non_zeros * beta + (1 - beta) * P

        return P_next * Z_next, [Z_next, P_next]


class TSB(IntermittentDemandModel):
    """Teunter,Syntetos and Babai (TSB) method using keras RNN layers.

    Using keras layers in anticipation of sharing information between time series.

    See https://www.lancaster.ac.uk/pg/waller/pdfs/Intermittent_Demand_Forecasting.pdf for model details.

    :param seasonality_type: string, type of multiplicative seasonality to use for all time series.
    """

    param_names = ["alpha", "beta", "Z0", "P0"]

    def __init__(self, seasonality=None, loss="mse"):
        self.x = None         # Need to save x for predicting out-of-sample
        self.y = None         # Need to save y for predicting out-of-sample
        self.rnnmodel = None

        self.loss = loss

        self.seasonality = seasonality

        self.model = None
        self.predictor = None

    def _create_model_predictor(self, lr):
        y = self.y
        Z0_start = y[:, 0]
        P0_start = (np.argmax(np.cumsum(y > 0, axis=1) == 2, axis=1)
                    - np.argmax(np.cumsum(y > 0, axis=1) == 1, axis=1)).reshape((y.shape[0], 1))

        inp_y = Input(shape=(None, 1))
        inp_emb_id = Input(shape=(1,))  # Dummy ID for embeddings

        if self.seasonality:
            inp_y_decoded = self.seasonality.apply_decoder(inp_y)
        else:
            inp_y_decoded = inp_y

        # (Ab)using embeddings here for initial value variables
        init_Z = Embedding(y.shape[0], 1, embeddings_initializer=constant(Z0_start), name="Z0")(inp_emb_id)[:, 0, :]
        init_P = Embedding(y.shape[0], 1, embeddings_initializer=constant(P0_start), name="P0")(inp_emb_id)[:, 0, :]

        rnn = RNN(TSBRNN(y.shape[0], input_shape=(y.shape[0],)), return_sequences=True, return_state=True)
        out_rnn = rnn((inp_y_decoded, tf.cast(inp_emb_id[:, None, :] * tf.ones_like(inp_y), tf.int32)),
                      initial_state=[init_Z, init_P])

        out = tf.keras.layers.concatenate([
            (init_Z * init_P)[:, :, None],
            out_rnn[0]
        ], axis=1)

        if self.seasonality:
            out = self.seasonality.apply_encoder(out)

        model = Model(inputs=[inp_y, inp_emb_id] + (self.seasonality.get_model_inputs() if self.seasonality else []),
                      outputs=out)
        model.compile(Adam(lr), self.loss)

        # predictor also outputs final state for predicting out-of-sample
        predictor = Model(
            inputs=[inp_y, inp_emb_id] + (self.seasonality.get_model_inputs() if self.seasonality else []),
            outputs=[out, out_rnn[1:]]
        )

        return model, predictor

    def __repr__(self):
        return "%s (%s)" % (
            self.__class__.__name__,
             ",".join(["%s=%s" % (k, np.round(v.ravel() if v.size > 0 else 0, 2)) for k, v in self.get_params().items()])
    )

