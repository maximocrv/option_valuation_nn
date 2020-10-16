import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from munch import munchify
import matplotlib.pyplot as plt

from EU_Option.data_generator import gen_model_data, gen_tf_model_data
from EU_Option.model_subclass import training_loop
from misc.heston_plotting import dual_plot, trend_coeff_r

with open('EU_Option/model_settings.json') as f:
    settings = json.load(f)

settings = munchify(settings)


def listdir_nohidden(path):
    for ele in os.listdir(path):
        if not ele.startswith('.'):
            yield ele


# PREPARING TRAINING DATA AND INITIATING MODEL
xy = pd.read_csv(settings.heston.paths.training_data, header=None)
x = xy.iloc[:, :-1]
x = np.array(x).astype(np.float32)
y = xy.iloc[:, -1]
y = np.array(y).astype(np.float32)

batch = 512
x_train, y_train, x_val, y_val, x_test, y_test = gen_model_data(x, y)
train_dataset, val_dataset, test_dataset = gen_tf_model_data(x_train, y_train, x_val, y_val, x_test, y_test, batch=batch)


# TRAINING LOOP
model = training_loop(train_dataset=train_dataset, val_dataset=val_dataset, epochs=2, batch=batch, input_dim=x.shape[1],
                      mode='u_T', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999))


num_grads = pd.read_csv(settings.heston.paths.num_put_grads, header=None)
num_grads.columns = ['u_S', 'u_T', 'u_K']


# for model in listdir_nohidden(settings.paths.BS_model_folder):
#     test_model = tf.keras.models.load_model(settings.paths.BS_model_folder + f'/{model}', compile=False)
#     fig = goodness_of_fit(model=test_model, numerical_grads=num_grads, x_test=x)
#     fig.savefig(settings.paths.plots_folder + f'/good_fit_{model}')
#     plt.close()

for model in listdir_nohidden(settings.paths.model_folder):
    test_model = tf.keras.models.load_model(settings.paths.model_folder + f'/{model}', compile=False)
    y_pred = test_model.call(x)
    coeffs, Rpara = trend_coeff_r(y, np.reshape(y_pred, (y_pred.shape[0],)))
    error = np.reshape(y, (y.shape[0], 1)) - y_pred
    fig = dual_plot(val=y, val_pred=y_pred, error=error, coeffs=coeffs, Rpara=Rpara)
    fig.savefig(settings.paths.plots_folder + f'/dual_plot_{model}')
    plt.close()
