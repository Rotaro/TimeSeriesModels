import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Embedding, RNN
from tensorflow.keras.initializers import constant
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam


from models.tf_model import AbsoluteMinMax, TFTimeSeriesModel


class Seasonality:
    """Shared seasonality for time series."""
    type_to_period = {
        "weekday": 7,
        "weekofyear": 53,
        "month": 12,
    }
    type_datetime_to_int_func = {
        "weekday": lambda dt: dt.weekday(),
        "weekofyear": lambda dt: dt.isocalendar()[1],
        "month": lambda dt: dt.month,
    }

    def __init__(self, seasonality_type="week"):
        assert seasonality_type in self.type_to_period, \
            "Invalid seasonality! Needs to be in %s!" % self.type_to_period.keys()

        self.seasonality_type = seasonality_type
        self.seasonal_period = self.type_to_period[self.seasonality_type]
        self._datetime_to_int_func = self.type_datetime_to_int_func[self.seasonality_type]

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
                name="seas_emb", embeddings_initializer=constant(0), embeddings_regularizer=l1_l2(l2=1e-3)
        )

    def _create_inputs(self):
        self._inputs = [
            Input(shape=(None,), name="inp_X_seas"),
            Input(shape=(None,), name="inp_Y_seas")
        ]

    def _create_decoder(self):
        self._decoder = self._embedding(self._inputs[0]) + 1

    def _create_encoder(self):
        self._encoder = self._embedding(self._inputs[1]) + 1

    def get_model_inputs(self):
        return self._inputs

    def get_fit_inputs(self, arr):
        return [arr[:, :-1], arr]

    def apply_decoder(self, apply_to):
        return apply_to / self._decoder

    def apply_encoder(self, apply_to):
        return apply_to * self._encoder

    def get_predict_inputs(self, arr):
        # Expand x by one for predicting one step ahead
        arr_expanded = (np.hstack([arr, arr[:, -1:] + 1]) % self.seasonal_period).astype(arr.dtype)

        return [arr, arr_expanded]

    def get_oos_predictions(self, first_oos_prediction, last_seasonality, n_oos_steps):
        x_oos = (last_seasonality[:, -1:] + np.arange(1, n_oos_steps + 1)) % self.seasonal_period

        deseason = self._embedding.get_weights()[0][x_oos[:, 0]] + 1
        oos_season = self._embedding.get_weights()[0][x_oos[:, 1:]][:, :, 0] + 1

        return first_oos_prediction / deseason * oos_season

    def get_weights(self):
        return self._embedding.get_weights()[0] + 1
