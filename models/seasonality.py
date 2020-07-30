import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.initializers import constant
from tensorflow.keras.regularizers import l1_l2


type_to_period = {
    "weekday": 7,
    "weekofyear": 53,
    "month": 12,
    "weekofmonth": 5,
}
type_datetime_to_int_func = {
    "weekday": lambda dt: dt.weekday(),
    "weekofyear": lambda dt: dt.isocalendar()[1] - 1,
    "month": lambda dt: dt.month - 1,
    "weekofmonth": lambda dt: dt.day // 7,
}


class SharedSeasonality:
    """Shared seasonality for time series."""
    def __init__(self, seasonality_type="week"):
        assert seasonality_type in type_to_period, \
            "Invalid seasonality! Needs to be in %s!" % type_to_period.keys()

        self.seasonality_type = seasonality_type
        self.seasonal_period = type_to_period[self.seasonality_type]
        self._datetime_to_int_func = type_datetime_to_int_func[self.seasonality_type]

        self._embedding = None
        self._inputs = None
        self._encoder = None
        self._decoder = None

        self._create_embedding()
        self._create_inputs()
        self._create_decoder()
        self._create_encoder()

    def datetime_to_array(self, arr):
        return np.vectorize(self._datetime_to_int_func)(arr)

    def _create_embedding(self):
        self._embedding = Embedding(
                self.seasonal_period, output_dim=1,
                name="seas_emb_%s" % self.seasonality_type, embeddings_initializer=constant(0),
                embeddings_regularizer=l1_l2(l2=1e-3)
        )

    def _create_inputs(self):
        self._inputs = [
            Input(shape=(None,), name="inp_X_seas_%s" % self.seasonality_type),
            Input(shape=(None,), name="inp_Y_seas_%s" % self.seasonality_type)
        ]

    def _create_decoder(self):
        self._decoder = self._embedding(self._inputs[0]) + 1

    def _create_encoder(self):
        self._encoder = self._embedding(self._inputs[1]) + 1

    def get_model_inputs(self):
        return self._inputs

    def get_fit_inputs(self, ids, x):
        return [x[:, :-1], x]

    def apply_decoder(self, apply_to):
        return apply_to / self._decoder

    def apply_encoder(self, apply_to):
        return apply_to * self._encoder

    def get_predict_inputs(self, ids, x):
        # Expand x by one for predicting one step ahead
        x_expanded = (np.hstack([x, x[:, -1:] + 1]) % self.seasonal_period).astype(x.dtype)

        return [x, x_expanded]

    def get_oos_predictions(self, first_oos_prediction, last_seasonality, n_oos_steps):
        x_oos = (last_seasonality[:, -1:] + np.arange(1, n_oos_steps + 1)) % self.seasonal_period

        deseason = self._embedding.get_weights()[0][x_oos[:, 0]] + 1
        oos_season = self._embedding.get_weights()[0][x_oos[:, 1:]][:, :, 0] + 1

        return first_oos_prediction / deseason * oos_season

    def get_weights(self):
        return self._embedding.get_weights()[0] + 1

    def plot_seasonality(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.get_weights())
        plt.title(self.seasonality_type, fontsize=15)


class FactorizedSeasonality:
    def __init__(self, n_time_series, n_dim=1, seasonality_type="weekday"):
        assert seasonality_type in type_to_period, \
            "Invalid seasonality! Needs to be in %s!" % type_to_period.keys()

        self.n_time_series = n_time_series
        self.n_dim = n_dim

        self.seasonality_type = seasonality_type
        self.seasonal_period = type_to_period[self.seasonality_type]
        self._datetime_to_int_func = type_datetime_to_int_func[self.seasonality_type]

        self._embedding = None
        self._inputs = None
        self._encoder = None
        self._decoder = None

        self._create_embedding()
        self._create_inputs()
        self._create_decoder()
        self._create_encoder()

    def datetime_to_array(self, arr):
        return np.vectorize(self._datetime_to_int_func)(arr)

    def _create_embedding(self):
        # First embedding for time series ID
        self._embedding = Embedding(
                self.n_time_series, output_dim=self.n_dim,
                name="seas_emb_%s" % self.seasonality_type,
                embeddings_initializer=constant(0), embeddings_regularizer=l1_l2(l2=1e-3)
        )

        self._seasonality_weights = Dense(self.seasonal_period, activation='linear')

    def _create_inputs(self):
        self._inputs = [
            Input(shape=(1,), name="timeseries_id_%s" % self.seasonality_type),
            Input(shape=(None,), name="inp_X_seas_%s" % self.seasonality_type, dtype=tf.int32),
            Input(shape=(None,), name="inp_Y_seas_%s" % self.seasonality_type, dtype=tf.int32)
        ]

    def _create_decoder(self):
        id_emb = self._embedding(self._inputs[0])[:, 0, :]
        id_seas_values = self._seasonality_weights(id_emb)
        seas_values = tf.gather(id_seas_values, self._inputs[1], axis=1, batch_dims=-1)

        self._decoder = seas_values[:, :, None] + 1

    def _create_encoder(self):
        id_emb = self._embedding(self._inputs[0])[:, 0, :]
        id_seas_values = self._seasonality_weights(id_emb)
        seas_values = tf.gather(id_seas_values, self._inputs[2], axis=1, batch_dims=-1)

        self._encoder = seas_values[:, :, None] + 1

    def get_model_inputs(self):
        return self._inputs

    def get_fit_inputs(self, ids, x):
        return [ids, x[:, :-1], x]

    def apply_decoder(self, apply_to):
        return apply_to / self._decoder

    def apply_encoder(self, apply_to):
        return apply_to * self._encoder

    def get_predict_inputs(self, ids, x):
        # Expand x by one for predicting one step ahead
        x_expanded = (np.hstack([x, x[:, -1:] + 1]) % self.seasonal_period).astype(x.dtype)

        return [ids, x, x_expanded]

    def get_oos_predictions(self, first_oos_prediction, last_seasonality, n_oos_steps):
        x_oos = (last_seasonality[:, -1:] + np.arange(1, n_oos_steps + 1)) % self.seasonal_period

        id_emb = self._embedding.get_weights()[0][np.arange(self.n_time_series)]
        id_seas_values = id_emb@self._seasonality_weights.get_weights()[0] + self._seasonality_weights.get_weights()[1]

        deseason = np.take_along_axis(id_seas_values, x_oos[:, :1], axis=1) + 1
        oos_season = np.take_along_axis(id_seas_values, x_oos[:, 1:], axis=1) + 1

        return first_oos_prediction / deseason * oos_season

    def get_weights(self):
        return self._embedding.get_weights()[0] @self._seasonality_weights.get_weights()[0] + \
               self._seasonality_weights.get_weights()[1] + 1

    def plot_seasonality(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.get_weights().T)
        plt.title(self.seasonality_type, fontsize=15)
