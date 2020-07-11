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

    params_to_plot = [
        {"trend": trend, "seasonal": seasonal, "damped": trend is not None}
        for trend in (None, "additive", "multiplicative",)
        for seasonal in (None, "additive", "multiplicative",)
    ]

    model_predictions = []
    for i_plot, params in enumerate(params_to_plot):
        # Exponential smoothing using keras RNN layers
        model_keras_rnn = expon_smoothing_tf.ExponentialSmoothing(**params, seasonal_period=seasonal_period)
        model_keras_rnn.fit(y[:-n_holdout], epochs=100, lr=1e-2)

        # Exponential smoothing using statsmodels implementation
        model_stats = ES_statsmodels(y[:-n_holdout], **params, seasonal_periods=seasonal_period).fit()
        model_predictions.append(
            (("ES - keras rnn %s" % str(params), model_keras_rnn.predict(n_holdout)),
             ("ES - statsmodels %s" % str(params), model_stats.predict(start=0, end=y.size - 1)))
        )

    # Plot keras rnn vs statmodels
    fig, axes = plt.subplots(len(params_to_plot) // 2 + (len(params_to_plot) % 2 == 1), 2, sharex=True)
    for i_plot, parameter_predictions in enumerate(model_predictions):
        ax = axes.ravel()[i_plot]
        ax.plot(df.time[:-n_holdout], y[:-n_holdout], "--", label="train")
        ax.plot(df.time[-n_holdout:], y[-n_holdout:], "--", label="test")

        for model_name, predictions in parameter_predictions:
            if np.isnan(predictions).sum() > 0:
                # Invalid values in predictions
                continue
            plot_and_calc_rmse(ax, df.time, df.value, predictions, n_holdout, model_name)
        ax.legend()
