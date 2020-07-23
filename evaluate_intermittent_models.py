import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.regression import mean_squared_error

from statsmodels.datasets import get_rdataset

import models.expon_smoothing_tf as expon_smoothing_tf
import models.crostons_method_tf as crostons_tf
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES_statsmodels


def plot_and_calc_rmse(ax, x, y_true, y_pred, n_holdout, label=None):
    ax.plot(x, y_pred, label="%s - RMSE: %.2f" % (
        label, np.sqrt(mean_squared_error(y_true[-n_holdout:], y_pred[-n_holdout:])))
    )


if __name__ == "__main__":
    plt.style.use("ggplot")

    n = 60

    # Some seasonality
    seasonality = np.array([0.8, 0.8, 0.9, 1.2, 1.5, 1.5])
    seasonality /= seasonality.sum() / seasonality.size
    # Just cyclical indices for now
    x = np.tile(np.arange(1, seasonality.size + 1), int(n // seasonality.size) + 1)[:n]
    x = np.tile(x, 3).reshape((3, n))

    # Some intermittent demand (=poisson process) with seasonality
    lam = np.array([0.1, 0.2, 0.8])
    y = np.random.poisson(lam[:, None] * x, (3, n))

    n_holdout = 12

    models = [
        (crostons_tf.CrostonsMethod(sba=False), {"epochs": 35, "lr": 1e-2}),
        (crostons_tf.CrostonsMethod(sba=True), {"epochs": 35, "lr": 1e-2}),
        (crostons_tf.TSB(), {"epochs": 35, "lr": 1e-2}),
        (crostons_tf.CrostonsMethod(sba=False, common_seasonality=True, loss="mse", seasonal_period=seasonality.size),
         {"epochs": 100, "lr": 1e-2, "x": x[:, :-n_holdout]}),
    ]
    fig, axes = plt.subplots(len(models) // 2 + (len(models) % 2 == 1), 2, sharex=True)
    for ax, (model, fit_opts) in zip(axes.ravel(), models):
        model.fit(y[:, :-n_holdout], **fit_opts)

        for i in range(y.shape[0]):
            ax.plot(y[i, :], "o--", label="actual values")
            plot_and_calc_rmse(ax, np.arange(y.shape[1]), y[i, :], model.predict(n_holdout)[i, :], n_holdout,
                               label="%s" % str(model))
        ax.legend(fontsize=8)
