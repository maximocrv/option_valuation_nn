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


def calc_arbitrage(_model, test_input):
    grads = y_x(_model, test_input)
    grads = pd.DataFrame(np.array(grads), columns=settings.grad_mat_cols)
    arbitrage_T = grads[grads.u_T <= 0]['u_T'].count()
    arbitrage_K = grads[grads.u_K >= 0]['u_K'].count()

    return arbitrage_T, arbitrage_K


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

    outputs = pd.read_csv(str(base_path / settings.paths.output_data), header=None,
                          names=settings.output_data_labels)

    val_inds = outputs.index[outputs.Vc >= 0.001]

    inputs = inputs.iloc[val_inds]
    x = np.array(inputs).astype(np.float32)

    outputs = outputs.iloc[val_inds]
    y_call = np.array(outputs.Vc).astype(np.float32)

    analytic_grads = pd.read_csv(str(base_path / settings.paths.true_grads), header=None,
                                 names=settings.true_grad_labels)
    analytic_grads = np.array(analytic_grads.iloc[val_inds])

    batch = 512
    random_state = 1

    x_train, y_train, x_val, y_val, x_test, y_test = gen_model_data(x, y_call, random_state=random_state)
    train_dataset, val_dataset, test_dataset = gen_tf_model_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                                                 batch=batch)

    grads_train, grads_test = train_test_split(analytic_grads, test_size=0.2, random_state=random_state)
    grads_train, grads_val = train_test_split(grads_train, test_size=0.25, random_state=random_state)

    lr_epochs = [30, 50, 65, 80]
    learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]

    _sc_weights = (10, 10)
    epsilon = 0
    # model_t, loss_t, val_loss_t = training_loop(train_dataset=train_dataset, val_dataset=val_dataset, epochs=20,
    #                                             batch=batch, input_dim=x.shape[1], mode='u_T',
    #                                             optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9,
    #                                                                                beta_2=0.999),
    #                                             sc_weights=_sc_weights, epsilon=epsilon)
    # arb_t_T, arb_t_K = calc_arbitrage(model_t, x_test)
    # grads_t = y_x(model_t, x_test)

    # model_k, loss_k, val_loss_k = training_loop(train_dataset=train_dataset, val_dataset=val_dataset, epochs=20,
    #                                             batch=batch, input_dim=x.shape[1], mode='u_K',
    #                                             optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9,
    #                                                                                beta_2=0.999),
    #                                             sc_weights=_sc_weights, epsilon=epsilon)
    # arb_k_T, arb_k_K = calc_arbitrage(model_k, x_test)
    # grads_k = y_x(model_k, x_test)

    model, loss, val_loss = training_loop(train_dataset=train_dataset, val_dataset=val_dataset, epochs=20,
                                          batch=batch, input_dim=x.shape[1], mode='mse',
                                          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9,
                                                                             beta_2=0.999),
                                          sc_weights=_sc_weights, epsilon=epsilon)
    arb_mse_T, arb_mse_K = calc_arbitrage(model, x_test)
    grads_mse = y_x(model, x_test)

    # sc_weights = [(1e-5, 1), (1e-4, 1), (1e-3, 1), (1e-2, 1), (1e-1, 1), (1, 1)]
    # epsilon = 1e-5
    # tracker = {}
    # for sc_weight in sc_weights:
    #     model, loss, val_loss = hybrid_training_loop(train_dataset=train_dataset, val_dataset=val_dataset, epochs=100,
    #                                                  batch=batch, input_dim=x.shape[1], mode='mse', lr_epochs=lr_epochs,
    #                                                  activation='relu', learning_rates=learning_rates,
    #                                                  sc_weights=sc_weight, epsilon=epsilon)
    #
        # arbt, arbk = calc_arbitrage(model, x_test)
        # test_pred = np.array(model(x_test))
        # val_loss_test = np.mean(np.abs(test_pred - y_test[..., np.newaxis]))
    #
    #     tracker[f'{sc_weight}'] = ['arbt: ', arbt, 'arbk: ', arbk, 'val_loss_test: ', val_loss_test]


    # model, loss, val_loss = hybrid_training_loop(train_dataset=train_dataset, val_dataset=val_dataset, epochs=100,
    #                                              batch=batch, input_dim=x.shape[1], mode='mse', lr_epochs=lr_epochs,
    #                                              activation='relu', learning_rates=learning_rates,
    #                                              sc_weights=(1e-4, 1), epsilon=epsilon)

    # model_folder_path = base_path / Path(settings.paths.model_folder)
    #
    # for model in listdir_nohidden(model_folder_path):
    #     # for model in training_settings:
    #     test_model = tf.keras.models.load_model(model_folder_path / Path(model), compile=False)
    #     # test_model = tf.keras.models.load_model(model_folder_path / Path(f'{model[0]}_{model[1]}'), compile=False)
    #     y_pred = test_model.call(x_test)
    #     error = np.mean(np.abs(y_test - np.reshape(y_pred, (y_pred.shape[0],))))
    #     print(f'{model}:{error}')

    # TODO: include weight parameter for the boiz and loop through to get the tings
    # model_path = base_path / Path(settings.paths.model_folder)
    # # for model in listdir_nohidden(model_path):
    # _model = 'u_KT_100'
    # test_model = tf.keras.models.load_model(model_path / Path(_model), compile=False)
    # #
    # print(f'{_model}: \n'
    #       f'arbitrage u_T: {arbitrage_T} \n'
    #       f'arbitrage u_K: {arbitrage_K}')
