from scipy import stats
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
from math import *
import pandas as pd
from operator import itemgetter
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

# I had to add these following two lines to prevent some weird errors, hopefully these won't be necessary if the code
# is properly fixed
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# tf.compat.v1.disable_eager_execution()

# Adjust your filepath to read the files here
df_train = pd.read_csv('/Users/Maximocravero/Documents/MATLAB/training_data_bss.csv', header=None)
df_train = np.array(df_train)
x_train_bs = df_train[:, [0, 1, 2, 3, 8]]

df_test = pd.read_csv('/Users/Maximocravero/Documents/MATLAB/test_data_bs.csv', header=None)
df_test = np.array(df_test)
x_test_bs = df_test[:, [0, 1, 2, 3, 8]]


def Black_Scholes_Greeks_Call(xarray):  # Calculating BS option values and Greeks
    S, T, K, r, v, = itemgetter(0, 1, 2, 3, 4)(xarray)
    T_sqrt = sqrt(T)
    d1 = (log(float(S) / K) + ((r) + v * v / 2.) * T) / (v * T_sqrt)
    d2 = d1 - v * T_sqrt
    Call_val = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    Delta = norm.cdf(d1)
    Gamma = norm.pdf(d1) / (S * v * T_sqrt)  # pg102 Higham
    Theta = -1 * (-(S * v * norm.pdf(d1)) / (2 * T_sqrt) - r * K * exp(-r * T) * norm.cdf(
        d2))  # *-1 due to dV/dt = -dV/dT!!
    Vega = S * T_sqrt * norm.pdf(d1)
    Rho = K * T * exp(-r * T) * norm.cdf(d2)
    Kderiv = -exp(-r * T) * norm.cdf(d2)

    return Call_val, Delta, Gamma, Theta, Vega, Rho, Kderiv


# Generating training/testing data, and theta for comparing with the NN


def gen_training_data(x_train_bs, x_test_bs):
    y_train_bs = []
    y_test_bs = []
    theta = []
    kderiv = []

    for i in range(len(x_train_bs)):
        theta.append(Black_Scholes_Greeks_Call(x_train_bs[i, :])[3])
        y_train_bs.append(Black_Scholes_Greeks_Call(x_train_bs[i, :])[0])
        kderiv.append(Black_Scholes_Greeks_Call(x_train_bs[i, :])[6])

    for j in range(len(x_test_bs)):
        y_test_bs.append(Black_Scholes_Greeks_Call(x_test_bs[j, :])[0])

    with open('y_train_bs.txt', 'w') as f:
        for item in y_train_bs:
            f.write(str(item) + "\n")

    with open('y_test_bs.txt', 'w') as f:
        for item in y_test_bs:
            f.write(str(item) + "\n")

    with open('../EU_Option/BS_Constraint/data/theta.txt', 'w') as f:
        for item in theta:
            f.write(str(item) + "\n")

    with open('../EU_Option/BS_Constraint/data/kderiv.txt', 'w') as f:
        for item in kderiv:
            f.write(str(item) + "\n")


# Only run the following line if you need to generate the training data, otherwise just use the following load_training_data
# function to load the .txt files
gen_training_data(x_train_bs, x_test_bs)
#
# def load_training_data():
#     y_train_bs = []
#     y_test_bs = []
#     theta = []
#     kderiv = []
#
#     with open("y_train_bs.txt", "r") as f:
#         for line in f:
#             y_train_bs.append(float(line.strip()))
#
#     with open("y_test_bs.txt", "r") as f:
#         for line in f:
#             y_test_bs.append(float(line.strip()))
#
#     with open("theta.txt", "r") as f:
#         for line in f:
#             theta.append(float(line.strip()))
#
#     with open("kderiv.txt", "r") as f:
#         for line in f:
#             kderiv.append(float(line.strip()))
#
#     return np.array(y_train_bs), np.array(y_test_bs), np.array(theta), np.array(kderiv)
#
#
# y_train_bs, y_test_bs, theta, kderiv = load_training_data()
#
#
# class MyModel(tf.keras.Model):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         #self.dense_in = Input(shape=(5,))
#         self.dense_in = Dense(5, activation='relu')
#         self.dense1 = Dense(400, activation='relu')
#         self.dense2 = Dense(400, activation='relu')
#         self.dense3 = Dense(400, activation='relu')
#         self.dense4 = Dense(400, activation='relu')
#         self.dense_out = Dense(1, activation='relu')
#         self.train_op = tf.keras.optimizers.Adagrad(learning_rate=0.1)
#
#     def call(self, inputs):
#         x = self.dense_in(inputs)
#         x = self.dense1(x)
#         x = self.dense2(x)
#         x = self.dense3(x)
#         x = self.dense4(x)
#         return self.dense_out(x)
#
#     def get_loss(self, X, Y):
#         boom = self.dense_in(X)
#         boom1 = self.dense1(boom)
#         boom2 = self.dense2(boom1)
#         boom3 = self.dense3(boom2)
#         boom4 = self.dense4(boom3)
#         boom5 = self.dense_out(boom4)
#         return tf.math.square(boom5-Y)
#
#     def get_grad(self, X, Y, model):
#         with tf.GradientTape() as tape:
#             tape.watch(self.dense1.variables)
#             tape.watch(self.dense2.variables)
#             tape.watch(self.dense3.variables)
#             tape.watch(self.dense4.variables)
#             tape.watch(self.dense_out.variables)
#             L = self.get_loss(X, Y)
#             g = tape.gradient(L, model.trainable_variables)
#         return g
#
#     def network_learn(self, X, Y):
#         g = self.get_grad(X, Y)
#         self.train_op.apply_gradients(zip(g, model.trainable_variables))
#
#     # def get_loss(self, x, y):
#     #     with tf.GradientTape() as tape:
#     #         tape.watch(self.dense1)
#     #         tape.watch(self.dense2)
#     #         tape.watch(self.dense3)
#     #         tape.watch(self.dense4)
#     #         tape.watch(self.dense_out)
#
#
# def custom_loss(model, y_true, y_pred):
#         K = model.input[2]
#         loss_value = tape.gradient(model.output, K) + tf.reduce_mean(tf.square(y_true - y_pred))
#         return loss_value, tape.gradient(loss_value, model.trainable_variables)
#
#
# model = MyModel()
#
# epochs = 10
# SIZE = x_test_bs.shape[0]
# batch = 100
# opt = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999)
#
# for i in range(epochs):
#     for j in range(0, SIZE, batch):
#         with tf.GradientTape() as tape:
#             #tape.watch(model.input)
#         #     loss_value, grads = custom_loss(model, y_train_bs[j:j + batch], model(y_train_bs[j:j + batch]))
#         # opt.apply_gradients(zip(grads, model.trainable_variables))
#             print('MyModel: ', i, ' : ', np.mean(loss_value.numpy()))
#
# # model = MyModel()
# # for i in range(100):
# #     model.network_learn(x_train_bs, y_train_bs)
#
