"""In this module we define functions for model gradients, plots containing histograms and goodness of fit lines."""
import os
import json
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from munch import Munch, munchify

from EU_Option.helpers import y_x, y_xx, ecdf, trend_coeff_r, listdir_nohidden, get_base_path


def dual_plot(val: np.array, val_pred: np.array, error: list, coeffs: np.poly1d, Rpara: str):
    x, y = ecdf(error)

    perf = plt.figure(figsize=(12, 6))
    ax1 = perf.add_subplot(121)
    ax1.text(0.1, 0.9, s=Rpara, transform=ax1.transAxes)
    ax1.scatter(val, val_pred, s=0.01, marker="x")
    ax1.set_xlabel('Test Values')
    ax1.set_ylabel('Predicted Values')
    ax1.plot(val, coeffs(val), c='k')
    ax1.set_title('European Option Values')

    ax2 = perf.add_subplot(122)
    ax2.hist(np.array(error), bins=50, density=True, color=None, histtype="stepfilled", alpha=0.8)
    ax2.set_title('Histogram of Errors')
    ax2.set_xlabel('Error')

    ax3 = ax2.twinx()
    ax3.scatter(x, y, s=0.5, marker='o', c='r')
    ax3.set_ylabel('F(x)')

    return perf


def goodness_of_fit(model: tf.keras.Model, numerical_grads: pd.DataFrame, x_test):
    grad_mat_i = y_x(model, x_test)
    grad_mat_ii = y_xx(model, x_test)
    grad_mat_i = pd.DataFrame(data=np.array(grad_mat_i), columns=['u_S', 'u_T', 'u_K', 'u_r', 'u_rho',
                                                                  'u_kap', 'u_gamma', 'u_v0', 'u_vbar'])

    coeffs_k, Rpara_k = trend_coeff_r(numerical_grads['u_K'], grad_mat_i['u_K'])
    coeffs_t, Rpara_t = trend_coeff_r(numerical_grads['u_T'], grad_mat_i['u_T'])

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax1.text(0.1, 0.9, s=Rpara_k, transform=ax1.transAxes)
    ax1.scatter(numerical_grads['u_K'], grad_mat_i['u_K'], s=0.01, marker="x")
    ax1.set_xlabel('Test Values')
    ax1.set_ylabel('Predicted Values')
    ax1.plot(numerical_grads['u_K'], coeffs_k(numerical_grads['u_K']), c='k')
    ax1.set_title('Line of Best Fit u_K')

    ax2 = fig.add_subplot(122)
    ax2.text(0.1, 0.9, s=Rpara_t, transform=ax2.transAxes)
    ax2.scatter(numerical_grads['u_T'], grad_mat_i['u_T'], s=0.01, marker="x")
    ax2.set_xlabel('Test Values')
    ax2.set_ylabel('Predicted Values')
    ax2.plot(numerical_grads['u_T'], coeffs_t(numerical_grads['u_T']), c='k')
    ax2.set_title('Line of Best Fit u_T')

    return fig


def gen_good_fit_plots(model_folder_path: Union[str, Path], plot_path: Union[str, Path],
                       num_grads: pd.DataFrame, x: np.array):
    root = get_base_path()
    data_path = root / plot_path
    data_path.mkdir(parents=True, exist_ok=True)
    for model in listdir_nohidden(model_folder_path):
        test_model = tf.keras.models.load_model(f'heston_model_folder/{model}', compile=False)
        fig = goodness_of_fit(model=test_model, numerical_grads=num_grads, x_test=x)
        fig.savefig(Path(plot_path + f'/good_fit_{model}'))
        plt.close()


def gen_dual_plots(model_folder_path: Union[str, Path], plot_path: Union[str, Path], x: np.array, y: np.array):
    root = get_base_path()
    data_path = root / plot_path
    data_path.mkdir(parents=True, exist_ok=True)
    for model in listdir_nohidden(model_folder_path):
        test_model = tf.keras.models.load_model(model_folder_path + f'/{model}', compile=False)

        y_pred = test_model.call(x)
        coeffs, Rpara = trend_coeff_r(y, np.reshape(y_pred, (y_pred.shape[0],)))
        error = np.reshape(y, (y.shape[0], 1)) - y_pred

        fig = dual_plot(val=y, val_pred=y_pred, error=error, coeffs=coeffs, Rpara=Rpara)
        fig.savefig(Path(plot_path + f'/dual_plot_{model}'))
        plt.close()


if __name__ == "__main__":
    base_path = get_base_path()

    with open(base_path / 'model_settings.json') as settings_file:
        settings = json.load(settings_file)
    settings = munchify(settings)