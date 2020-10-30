import numpy as np

import tensorflow as tf


def y_x(model, x):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as t:
        t.watch(x)
        _y = model.call(x)
        dy_dx = t.gradient(_y, x)
    return dy_dx


def y_xx(model, x):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as t:
        with tf.GradientTape() as t2:
            t2.watch(x)
            _y = model.call(x)
            dy_dx = t2.gradient(_y, x)
        d2y_dx2 = t.gradient(dy_dx, x)
    return d2y_dx2


def ecdf(data):
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return x, y


def trend_coeff_r(val, val_pred):
    trendline = np.polyfit(val, val_pred, 1)
    coeffs = np.poly1d(trendline)  # NOTE THAT THIS IS A LINE OF BEST FIT, MIGHT BE BEST TO JUST PLOT y=x FOR R^2 SCORE
    R = round(np.sum((coeffs(val) - np.mean(val_pred)) ** 2) / np.sum((val_pred - np.mean(val_pred)) ** 2), 4)
    Rpara = f"R^2 = {R}"
    return coeffs, Rpara
