import datetime as dt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.regression import mean_squared_error

from statsmodels.datasets import get_rdataset

import models.expon_smoothing_tf as expon_smoothing_tf
import models.crostons_method_tf as crostons_tf
from models.seasonality import SharedSeasonality, FactorizedSeasonality


def plot_and_calc_rmse(ax, x, y_true, y_pred, n_holdout, label=None):
    ax.plot(x, y_pred, label="%s - RMSE: %.2f" % (
        label, np.sqrt(mean_squared_error(y_true[-n_holdout:], y_pred[-n_holdout:])))
    )


if __name__ == "__main__":
    plt.style.use("ggplot")

    n_time = 120
    n_timeseries = 4

    # Some weekly seasonality
    seasonality = np.array([0.8, 0.8, 0.9, 0.9, 0.9, 2.5, 2.0])
    seasonality /= seasonality.sum() / seasonality.size

    # Some week of month seasonality
    week_seasonality = np.array([1.0, 1.5, 4.0, 1.5, 1.0])
    week_seasonality /= week_seasonality.sum() / week_seasonality.size

    x_dates = np.array([[dt.date(2019, 12, 30) + dt.timedelta(days=i) for i in range(n_time)]])
    x_dates = np.tile(x_dates, n_timeseries).reshape((n_timeseries, n_time))

    x_weekdays = np.vectorize(lambda x: x.weekday())(x_dates)
    x_week = np.vectorize(lambda x: x.day // 7)(x_dates)

    # Some intermittent demand (=poisson process) with seasonality
    x_seasonality = seasonality[x_weekdays]
    x_week_seasonality = week_seasonality[x_week]
    lam = np.linspace(0.1, 1.5, n_timeseries)[:, None]
    y = np.random.poisson(lam * x_seasonality * x_week_seasonality, (n_timeseries, n_time))

    n_holdout = 12

    models = [
        (crostons_tf.CrostonsMethod(sba=False), {"epochs": 35, "lr": 1e-2}, "Croston"),
        (crostons_tf.CrostonsMethod(sba=True), {"epochs": 35, "lr": 1e-2}, "Croston - SBA"),
        (crostons_tf.TSB(), {"epochs": 35, "lr": 1e-2}, "TSB"),
        (crostons_tf.CrostonsMethod(sba=False, seasonality=SharedSeasonality("weekday"), loss="mse"),
         {"epochs": 100, "lr": 1e-2, "x": x_dates[:, :-n_holdout]}, "Croston - Shared weekday"),
        (crostons_tf.CrostonsMethod(sba=False,
                                    seasonality=[SharedSeasonality("weekofmonth"), SharedSeasonality("weekday")],
                                    loss="mse"),
         {"epochs": 100, "lr": 1e-2, "x": x_dates[:, :-n_holdout]}, "Croston - Shared weekday + weekofmonth"),
        (crostons_tf.CrostonsMethod(sba=False,
                                    seasonality=FactorizedSeasonality(y.shape[0], n_dim=1, seasonality_type="weekday"),
                                    loss="mse"),
         {"epochs": 100, "lr": 1e-2, "x": x_dates[:, :-n_holdout]}, "Croston - Factorized Weekday"),
        (crostons_tf.CrostonsMethod(sba=False,
                                    seasonality=[
                                        FactorizedSeasonality(y.shape[0], n_dim=1, seasonality_type="weekday"),
                                        FactorizedSeasonality(y.shape[0], n_dim=1, seasonality_type="weekofmonth")
                                    ],
                                    loss="mse"),
         {"epochs": 100, "lr": 1e-2, "x": x_dates[:, :-n_holdout]}, "Croston - Factorized weekday + weekofmonth"),
        (crostons_tf.TSB(seasonality=SharedSeasonality("weekday"), loss="mse"),
         {"epochs": 100, "lr": 1e-2, "x": x_dates[:, :-n_holdout]}, "TSB - Shared weekday"),
    ]
    fig, axes = plt.subplots(len(models) // 2 + (len(models) % 2 == 1), 2, sharex=True)
    for ax, (model, fit_opts, title) in zip(axes.ravel(), models):
        model.fit(y[:, :-n_holdout], **fit_opts)

        ax.vlines(np.arange(y.shape[1])[-n_holdout], y.min(), y.max(), linestyles="dashed")
        for i in range(y.shape[0]):
            ax.plot(y[i, :], "o--", label="actual values")
            plot_and_calc_rmse(ax, np.arange(y.shape[1]), y[i, :], model.predict(n_holdout)[i, :], n_holdout,
                               label="%s" % str(model))
            ax.set_title(title)
        ax.legend(fontsize=8)
