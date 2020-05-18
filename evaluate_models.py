import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.regression import mean_squared_error

from statsmodels.datasets import get_rdataset

import models.expon_smoothing_tf as expon_smoothing_tf
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES_statsmodels


def plot_and_calc_rmse(ax, x, y_true, y_pred, n_holdout, label=None):

    ax.plot(x, y_pred, label="%s - RMSE: %.2f" % (
        label, np.sqrt(mean_squared_error(y_true[-n_holdout:], y_pred[-n_holdout:]))
    ))


if __name__ == "__main__":
    plt.style.use("ggplot")

    # Evaluate models on last year of AirPassengers dataset
    df = get_rdataset("AirPassengers").data
    y = df.value

    n_holdout = 24
    seasonal_period = 12

    predictions = []

    # Exponential smoothing using keras RNN layers
    model_keras_rnn = expon_smoothing_tf.ExponentialSmoothing(seasonal_period=seasonal_period)
    model_keras_rnn.fit(y[:-n_holdout], epochs=100, lr=1e-2)
    predictions.append(("ES - keras RNN", model_keras_rnn.predict(n_holdout)))

    # Exponential smoothing using keras RNN layers
    model_keras_rnn_add = expon_smoothing_tf.ExponentialSmoothing(season="additive", seasonal_period=seasonal_period)
    model_keras_rnn_add.fit(y[:-n_holdout], epochs=100, lr=1e-2)
    predictions.append(("ES - keras RNN additive z", model_keras_rnn_add.predict(n_holdout)))

    # Exponential smoothing using statsmodels implementation
    model_stats = ES_statsmodels(y[:-n_holdout], trend="additive", damped=True, seasonal="multiplicative",
                                 seasonal_periods=seasonal_period).fit()
    predictions.append(("ES - statsmodels", model_stats.predict(start=0, end=y.size - 1)))

    # Exponential smoothing using statsmodels implementation
    model_stats_add = ES_statsmodels(y[:-n_holdout], trend="additive", damped=True, seasonal="additive",
                                     seasonal_periods=seasonal_period).fit()
    predictions.append(("ES - statsmodels additive", model_stats_add.predict(start=0, end=y.size - 1)))

    plt.figure()
    plt.plot(df.time[:-n_holdout], y[:-n_holdout], "--", label="train")
    plt.plot(df.time[-n_holdout:], y[-n_holdout:], "--", label="test")
    ax = plt.gca()
    for model_name, model_predictions in predictions:
        plot_and_calc_rmse(plt.gca(), df.time, df.value, model_predictions, n_holdout, model_name)
    plt.legend()
