import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from munch import munchify

from EU_Option.train_helpers import training_loop
from EU_Option.helpers import get_base_path, listdir_nohidden
from EU_Option.data_generator import gen_model_data, gen_tf_model_data

if __name__ == '__main__':
    base_path = get_base_path()
    with open(base_path / 'model_settings.json') as f:
        settings = json.load(f)
    settings = munchify(settings)
    misc_settings = settings

    pde_mode = 'black_scholes'

    if pde_mode == 'black_scholes':
        settings = settings.bs
    elif pde_mode == 'heston':
        settings = settings.heston
    else:
        print('Please enter a valid PDE mode')

    # PREPARING TRAINING DATA AND INITIATING MODEL
    inputs = pd.read_csv(str(base_path / settings.paths.input_data), header=None,
                         names=settings.input_data_labels)
    inputs = inputs.drop(['S'], axis=1)
    x = np.array(inputs).astype(np.float32)

    outputs = pd.read_csv(str(base_path / settings.paths.output_data), header=None,
                          names=settings.output_data_labels)
    y_call = np.array(outputs.Vc).astype(np.float32)

    analytic_grads = pd.read_csv(str(base_path / settings.paths.true_grads), header=None,
                                 names=settings.true_grad_labels)

    batch = 512
    x_train, y_train, x_val, y_val, x_test, y_test = gen_model_data(x, y_call)
    train_dataset, val_dataset, test_dataset = gen_tf_model_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                                                 batch=batch)

    # for mode, epoch in product(misc_settings.training_modes, misc_settings.epochs):
    training_settings = [('u_KT', 5), ('u_KT', 10), ('u_KT', 20), ('u_KT', 50),
                          ('u_T', 5), ('u_T', 10), ('u_T', 20), ('u_T', 50), ('u_T', 100),
                         ('u_K', 2), ('u_K', 5), ('u_K', 10), ('u_K', 20), ('u_K', 50), ('u_K', 100)]

    # sep_list = [('mse', 20), ('mse', 10), ('u_K', 2)]
    # for mode, epoch in sep_list:
    #     name = mode + f'_{epoch}'
    #     model, loss, val_loss = training_loop(train_dataset=train_dataset, val_dataset=val_dataset, epochs=epoch,
    #                                           batch=batch, input_dim=x.shape[1], mode=mode,
    #                                           optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9,
    #                                                                              beta_2=0.999))
    #
    #     loss = np.array(loss)
    #     loss = pd.DataFrame(loss)
    #     loss.to_csv(base_path / settings.paths.losses / f'loss_{name}.csv')
    #
    #     val_loss = np.array(val_loss)
    #     val_loss = pd.DataFrame(val_loss)
    #     val_loss.to_csv(base_path / settings.paths.losses / f'val_loss_{name}.csv')
    #
    #     caller = np.random.uniform(0, 2, (1, x.shape[1])).astype(np.float32)
    #     model(caller)
    #     model.save(base_path / settings.paths.model_folder / f'{name}')

    model_folder_path = base_path / Path(settings.paths.model_folder)
    for model in listdir_nohidden(model_folder_path):
    # for model in training_settings:
        test_model = tf.keras.models.load_model(model_folder_path / Path(model), compile=False)
        # test_model = tf.keras.models.load_model(model_folder_path / Path(f'{model[0]}_{model[1]}'), compile=False)
        y_pred = test_model.call(x_test)
        error = np.mean(np.abs(y_test - np.reshape(y_pred, (y_pred.shape[0],))))
        print(f'{model}:{error}')
