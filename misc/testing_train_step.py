import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, InputLayer, Input
from EU_Call.BS_Constraint.data_generator_module import load_xy
from EU_Call.BS_Constraint.plotting_functions_bs import y_x, y_xx, ecdf, dual_plot, trend_coeff_r
from EU_Call.BS_Constraint.data_generator_module import BS_Call, gen_training_data, load_training_data
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop


x_path = '/Users/Maximocravero/Documents/MATLAB/x_training_data_bs_ii.csv'
y_path = '/Users/Maximocravero/Documents/MATLAB/y_training_data_bs_ii.csv'

x, y = load_xy(x_path=x_path, y_path=y_path)

batch = 256

## DEFINING MODEL
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        #self.dense_in = InputLayer(input_shape=(5,))
        #self.dense_in = Input(shape=(5,))
        self.dense1 = Dense(400, activation='relu', kernel_initializer=initializers.glorot_uniform(), input_dim=5)
        self.dense2 = Dense(400, activation='relu', kernel_initializer=initializers.glorot_uniform())
        self.dense3 = Dense(400, activation='relu', kernel_initializer=initializers.glorot_uniform())
        self.dense4 = Dense(400, activation='relu', kernel_initializer=initializers.glorot_uniform())
        self.dense_out = Dense(1, activation='relu', kernel_initializer=initializers.glorot_uniform())

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 5), dtype=tf.float32, name='inputs')])   #CHECK tf.saved_model.save docs!
    # @tf.function
    def call(self, inputs, **kwargs):
        #x = self.dense_in(inputs)
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense_out(x)

    # @tf.function
    def get_loss(self, X, Y):
        with tf.GradientTape() as tape:
            tape.watch(tf.convert_to_tensor(X))
            Y_pred = self.call(X)
            # u_K = tf.redu
            # u_S = tf.reduceelkrgn;earg
        return tf.reduce_mean(tf.math.square(Y_pred-Y)) + tf.reduce_mean(tf.maximum(0, tape.gradient(Y_pred, X)[:, 2]))
        # return tf.reduce_mean(tf.math.square(Y_pred-Y))

    # @tf.function
    def get_grad_and_loss(self, X, Y):
        with tf.GradientTape() as tape:
            tape.watch(tf.convert_to_tensor(X))
            L = self.get_loss(X, Y)
        g = tape.gradient(L, self.trainable_weights)
        return g, L

    def train_step(self, data):
        # data = data_adapter.expand_1d(data)
        # x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        #
        # y_pred = self(x, training=True)
        # # loss = self.compiled_loss(
        # #     y, y_pred, sample_weight, regularization_losses=self.losses)
        # # gradients = tape.gradient(loss, self.trainable_variables)
        # # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        #
        # grads, L = model.get_grad_and_loss(x, y)
        # optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #
        # self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # return {m.name: m.result() for m in self.metrics}
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            # loss = self.compiled_loss(
            #     y, y_pred, sample_weight, regularization_losses=self.losses)
        # For custom training steps, users can just write:
        #   trainable_variables = self.trainable_variables
        #   gradients = tape.gradient(loss, trainable_variables)
        #   self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # The _minimize call does a few extra steps unnecessary in most cases,
        # such as loss scaling and gradient clipping.
            grads, L = self.get_grad_and_loss(x, y)
            optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


## INITIATING MODEL AND PREPARING TRAINING DATA
model = MyModel()
# model._set_inputs(model.dense_in)
epochs = 1
# SIZE = x_train_bs.shape[0]
# x_train_bs = x_train_bs[np.newaxis, :]
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
# val_batch = 1
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch)

model.compile(optimizer='Adam', metrics=['mae'])

model.fit(x_train, y_train, batch_size=batch, epochs=5, validation_data=(x_val, y_val), verbose=1)
