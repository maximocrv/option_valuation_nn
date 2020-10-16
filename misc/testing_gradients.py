from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
from keras import initializers
from keras import optimizers
import pandas as pd
import tensorflow as tf


df_train = pd.read_excel(r'C:\Users\maxic\Desktop\Finance Research\European Call NN\training_data.xlsx', header = None)
df_test = pd.read_excel(r'C:\Users\maxic\Desktop\Finance Research\European Call NN\test_data.xlsx', header = None)

x_train = df_train[df_train.columns[0:8]]
y_train = df_train[df_train.columns[8]]

x_test = df_test[df_test.columns[0:8]]
y_test = df_test[df_test.columns[8]]

input_shape = (8,)

# model = Sequential()
# initializers.glorot_uniform(seed = None)
# model.add(Dense(400, activation = 'relu', input_shape = input_shape))
# model.add(Dense(400, activation = 'relu'))
# model.add(Dense(400, activation = 'relu'))
# model.add(Dense(400, activation = 'relu'))
# model.add(Dense(1, activation = 'relu'))
#
#
# def newloss(y,yhat):
#     msee = K.mean(K.square(y-yhat))
#
#     #grads = tf.GradientTape(y_train, x_train[1])
#     #constrloss = msee + max(-grads,0)
#     return msee
#
# #Time to maturity (T) is in second column i.e. index 1 in python
#
# sgd = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
# model.compile(loss = 'mse',
#               optimizer = sgd,
#               metrics = ['mae'])
#
# epochs = 5
# batch_size = 1024
# # Fit the model weights.
# history = model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
#
# model.save('heston_nn.h5')

## Then load it by loaded_model()
file_heston = 'heston_nn.h5'
loaded_model = keras.models.load_model(file_heston)


weights = loaded_model.weights  # weight tensors
gradients = loaded_model.optimizer.get_gradients(loaded_model.total_loss, weights)  # gradient tensors

input_tensors = [*loaded_model.inputs,
                 loaded_model.sample_weights[0],  # sample weights
                 loaded_model.targets[0],  # labels
                 K.learning_phase()]  # train or test mode]

x_train = pd.DataFrame(x_train).to_numpy()
y_train = pd.DataFrame(y_train).to_numpy()

input_tensors = [x_train,
          np.ones(x_train.shape[0]),
          y_train,  # y labels
          0]

get_gradients = K.function(inputs = input_tensors, outputs = gradients)



sample_lens = len(x_train[0])
print('the number of samples is ', sample_lens)
# inputs = [np.reshape(x_train[0], (sample_lens, 1)),  # X input data, including x0=Stock, x1=Time, .....
#           np.reshape(x_train[1], (sample_lens, 1)),
#           np.reshape(x_train[2], (sample_lens, 1)),
#           np.reshape(x_train[3], (sample_lens, 1)),
#           np.reshape(x_train[4], (sample_lens, 1)),
#           np.reshape(x_train[5], (sample_lens, 1)),
#           np.reshape(x_train[6], (sample_lens, 1)),
#           np.reshape(x_train[7], (sample_lens, 1)),
#           np.ones((sample_lens,)),
#           y_train,
#           0
#           ]

# inputs = [x_train[0].values.reshape((sample_lens, 1)),
#           x_train[1].values.reshape((sample_lens, 1)),
#           x_train[2].values.reshape((sample_lens, 1)),
#           x_train[3].values.reshape((sample_lens, 1)),
#           x_train[4].values.reshape((sample_lens, 1)),
#           x_train[5].values.reshape((sample_lens, 1)),
#           x_train[6].values.reshape((sample_lens, 1)),
#           x_train[7].values.reshape((sample_lens, 1)),
#           np.ones((sample_lens)),
#           y_train,
#           0
#           ]
#
gradient_matrix = get_gradients(inputs)