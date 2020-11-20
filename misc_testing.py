import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from munch import munchify
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from EU_Option.tools import y_x
from EU_Option.train_helpers import adaptive_training_loop, training_loop, hybrid_training_loop
from EU_Option.helpers import get_base_path, listdir_nohidden
from EU_Option.data_generator import gen_model_data, gen_tf_model_data
from EU_Option.model_plotting import goodness_of_fit, dual_plot


def loss_plot(loss, val_loss):
    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.plot(loss)
    ax1.set_xlabel('Batches')
    ax1.set_ylabel('Loss')
    ax1.set_title('Objective Function Evolution')

    ax2 = fig.add_subplot(122)
    ax2.plot(val_loss)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss per Epoch')

    return fig


if __name__ == '__main__':
    base_path = get_base_path()
    with open(base_path / 'model_settings.json') as f:
        settings = json.load(f)
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

    analytic_grads = pd.read_csv(str(base_path / settings.paths.true_grads), header=None,
                                 names=settings.true_grad_labels)
    analytic_grads = np.array(analytic_grads)

    batch = 256
    random_state = 1

    x_train, y_train, x_val, y_val, x_test, y_test = gen_model_data(x, y_call, random_state=random_state)
    train_dataset, val_dataset, test_dataset = gen_tf_model_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                                                 batch=batch)

    grads_train, grads_test = train_test_split(analytic_grads, test_size=0.2, random_state=random_state)
    grads_train, grads_val = train_test_split(grads_train, test_size=0.25, random_state=random_state)

    lr_epochs = [30, 50, 65, 80]
    learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]
    # misc_testing(9) 1e-2 grad constraint factor, misc_testing(10) 1e-3 grad constraint factor
    model, loss, val_loss = hybrid_training_loop(train_dataset=train_dataset, val_dataset=val_dataset, epochs=100,
                                                 batch=batch, input_dim=x.shape[1], mode='mse', lr_epochs=lr_epochs,
                                                 activation='relu', learning_rates=learning_rates)


    # model_path = base_path / Path(settings.paths.model_folder)
    # for model in listdir_nohidden(model_path):
    #     test_model = tf.keras.models.load_model(model_path / Path(model), compile=False)
    #     grads = y_x(test_model, x_test)
    #     grads = pd.DataFrame(np.array(grads), columns=settings.grad_mat_cols)
    #     arbitrage_T = grads[grads.u_T <= 0]['u_T'].count()
    #     arbitrage_K = grads[grads.u_K >= 0]['u_K'].count()
    #
    #     print(f'{model}: \n'
    #           f'arbitrage u_T: {arbitrage_T} \n'
    #           f'arbitrage u_K: {arbitrage_K}')
