"""Define functions for generating model gradients, plots containing histograms and goodness of fit lines."""
import os
import json
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from munch import Munch, munchify
from sklearn.model_selection import train_test_split

from EU_Option.data_generator import gen_model_data
from EU_Option.helpers import listdir_nohidden, get_base_path
from EU_Option.tools import y_x, y_xx, ecdf, trend_coeff_r


def dual_plot(val: np.array, val_pred: np.array, error: list, coeffs: np.poly1d, Rpara: str):
    x_cum, y_cum = ecdf(error)

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
    ax3.scatter(x_cum, y_cum, s=0.5, marker='o', c='r')
    ax3.set_ylabel('F(x)')

    return perf


def goodness_of_fit(model: tf.keras.Model, true_grads: pd.DataFrame, x_test, grad_mat_cols: list):
    model_grads_i = y_x(model, x_test)
    model_grads_ii = y_xx(model, x_test)
    model_grads_i = pd.DataFrame(data=np.array(model_grads_i), columns=grad_mat_cols)

    coeffs_k, Rpara_k = trend_coeff_r(true_grads['dCdK'], model_grads_i['u_K'])
    coeffs_t, Rpara_t = trend_coeff_r(true_grads['dCdT'], model_grads_i['u_T'])

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.text(0.1, 0.9, s=Rpara_k, transform=ax1.transAxes)
    ax1.scatter(true_grads['dCdK'], model_grads_i['u_K'], s=0.01, marker="x")
    ax1.set_xlabel('Test Values')
    ax1.set_ylabel('Predicted Values')
    ax1.plot(true_grads['dCdK'], coeffs_k(true_grads['dCdK']), c='k')
    ax1.set_title('Line of Best Fit dCdK')

    ax2 = fig.add_subplot(122)
    ax2.text(0.1, 0.9, s=Rpara_t, transform=ax2.transAxes)
    ax2.scatter(true_grads['dCdT'], model_grads_i['u_T'], s=0.01, marker="x")
    ax2.set_xlabel('Test Values')
    ax2.set_ylabel('Predicted Values')
    ax2.plot(true_grads['dCdT'], coeffs_k(true_grads['dCdT']), c='k')
    ax2.set_title('Line of Best Fit dCdT')

    return fig


def gen_dual_plots(model_folder_path: Union[str, Path], plot_folder: Union[str, Path], x: np.array, y: np.array):
    plot_folder.mkdir(parents=True, exist_ok=True)

    for model in listdir_nohidden(model_folder_path):
        test_model = tf.keras.models.load_model(model_folder_path / Path(model), compile=False)

        y_pred = test_model.call(x)
        y_pred = np.reshape(y_pred, (y_pred.shape[0],))
        coeffs, Rpara = trend_coeff_r(y, y_pred)
        error = y_pred - y

        fig = dual_plot(val=y, val_pred=y_pred, error=error, coeffs=coeffs, Rpara=Rpara)
        fig.savefig(plot_folder / Path(f'dual_plot_{model}'))
        plt.close()


def gen_good_fit_plots(model_folder_path: Union[str, Path], plot_folder: Union[str, Path],
                       grads: pd.DataFrame, x: np.array, grad_mat_cols: list):
    plot_folder.mkdir(parents=True, exist_ok=True)

    for model in listdir_nohidden(model_folder_path):
        test_model = tf.keras.models.load_model(model_folder_path / Path(model), compile=False)
        fig = goodness_of_fit(model=test_model, true_grads=grads, x_test=x, grad_mat_cols=grad_mat_cols)
        fig.savefig(plot_folder / Path(f'good_fit_{model}'))
        plt.close()


if __name__ == "__main__":
    base_path = get_base_path()

    with open(base_path / 'model_settings.json') as settings_file:
        settings = json.load(settings_file)
    settings = munchify(settings)

    pde_mode = 'black_scholes'

    if pde_mode == 'black_scholes':
        settings = settings.bs
    elif pde_mode == 'heston':
        settings = settings.heston
    else:
        print('Please enter a valid PDE mode')

    inputs = pd.read_csv(str(base_path / settings.paths.input_data), header=None,
                         names=settings.input_data_labels)
    inputs = inputs.drop(['S'], axis=1)
    x = np.array(inputs).astype(np.float32)

    outputs = pd.read_csv(str(base_path / settings.paths.output_data), header=None,
                          names=settings.output_data_labels)
    y_call = np.array(outputs.Vc).astype(np.float32)
    x_train, y_train, x_val, y_val, x_test, y_test = gen_model_data(x, y_call)

    true_grads = pd.read_csv(str(base_path / settings.paths.true_grads), header=None,
                             names=settings.true_grad_labels)
    analytic_grads = np.array(true_grads)
    random_state = 1
    grads_train, grads_test = train_test_split(analytic_grads, test_size=0.2, random_state=random_state)
    grads_train, grads_val = train_test_split(grads_train, test_size=0.25, random_state=random_state)
    grads_test = pd.DataFrame(grads_test, columns=settings.true_grad_labels)

    model_path = base_path / Path(settings.paths.model_folder)
    dual_plot_folder = base_path / Path(settings.paths.plots_folder + '/dual_plots')
    good_fit_plot_folder = base_path / Path(settings.paths.plots_folder + '/good_fit')

    # gen_dual_plots(model_path, dual_plot_folder, x_test, y_test)

    # grad_mat_cols = settings.grad_mat_cols
    # gen_good_fit_plots(model_path, good_fit_plot_folder, grads_test, x_test, grad_mat_cols)

    val_loss_mse_100 = pd.read_csv('/Users/Maximocravero/PycharmProjects/Finance Research/EU_Option/BS_Constraint/Losses/val_loss_mse_100.csv')
    val_loss_mse_100 = val_loss_mse_100.iloc[:, 1]

    val_loss_uk_100 = pd.read_csv('/Users/Maximocravero/PycharmProjects/Finance Research/EU_Option/BS_Constraint/Losses/val_loss_u_K_100.csv')
    val_loss_uk_100 = val_loss_uk_100.iloc[:, 1]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(np.arange(0, len(val_loss_mse_100)), np.log(val_loss_mse_100))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Log Validation Loss')
    ax1.set_title('Validation Loss for MSE objective')

    ax2 = fig.add_subplot(122)
    ax2.plot(np.arange(0, len(val_loss_uk_100)), np.log(val_loss_uk_100))
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Log Validation Loss')
    ax2.set_title('Validation Loss for MSE and u_K objective')