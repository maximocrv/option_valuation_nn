from scipy.stats import norm
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
import matplotlib.pyplot as plt
import os

# I had to add these following two lines to prevent some weird errors, hopefully these won't be necessary if the code
# is properly fixed
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# tf.compat.v1.disable_eager_execution()

# Adjust your filepath to read the files here
df_train = pd.read_csv('/Users/Maximocravero/Documents/MATLAB/training_data_bs.csv', header = None)
df_train = np.array(df_train)
x_train_bs = df_train[:, [0, 1, 2, 3, 8]]

df_test = pd.read_csv('/Users/Maximocravero/Documents/MATLAB/test_data_bs.csv', header = None)
df_test = np.array(df_test)
x_test_bs = df_test[:, [0, 1, 2, 3, 8]]


def Black_Scholes_Greeks_Call(xarray): #Calculating BS option values and Greeks
    S, T, K, r, v, = itemgetter(0, 1, 2, 3, 4)(xarray)
    T_sqrt = sqrt(T)
    d1 = (log(float(S) / K) + ((r) + v * v / 2.) * T) / (v * T_sqrt)
    d2 = d1 - v * T_sqrt
    Call_val = S * norm.cdf(d1)-K * exp(-r * T) * norm.cdf(d2)
    Delta = norm.cdf(d1)
    Gamma = norm.pdf(d1) / (S * v * T_sqrt)  # pg102 Higham
    Theta = -1*(-(S * v * norm.pdf(d1)) / (2 * T_sqrt) - r * K * exp(-r * T) * norm.cdf(d2)) #*-1 due to dV/dt = -dV/dT!!
    Vega = S * T_sqrt * norm.pdf(d1)
    Rho = K * T * exp(-r * T) * norm.cdf(d2)
    Kderiv = -exp(-r * T) * norm.cdf(d2)
    
    return Call_val, Delta, Gamma, Theta, Vega, Rho, Kderiv
    
#Generating training/testing data, and theta for comparing with the NN


def gen_training_data(x_train_bs, x_test_bs):

    y_train_bs = []
    y_test_bs = []
    theta = []
    kderiv = []

    for i in range(len(x_train_bs)):
        theta.append(Black_Scholes_Greeks_Call(x_train_bs[i, :])[3])
        y_train_bs.append(Black_Scholes_Greeks_Call(x_train_bs[i, :])[0])
        y_test_bs.append(Black_Scholes_Greeks_Call(x_test_bs[i, :])[0])
        kderiv.append(Black_Scholes_Greeks_Call(x_test_bs[i, :])[6])

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
# gen_training_data(x_train_bs, x_test_bs)

def load_training_data():

    y_train_bs = []
    y_test_bs = []
    theta = []
    kderiv = []

    with open("BS_Constraint/y_train_bs.txt", "r") as f:
        for line in f:
            y_train_bs.append(float(line.strip()))

    with open("BS_Constraint/y_test_bs.txt", "r") as f:
        for line in f:
            y_test_bs.append(float(line.strip()))

    with open("BS_Constraint/theta.txt", "r") as f:
        for line in f:
            theta.append(float(line.strip()))

    with open("BS_Constraint/kderiv.txt", "r") as f:
        for line in f:
            kderiv.append(float(line.strip()))

    return np.array(y_train_bs), np.array(y_test_bs), np.array(theta), np.array(kderiv)


y_train_bs, y_test_bs, theta, kderiv = load_training_data()

#Defining the NN, relu/tanh/softplus/sigmoid
input_shape = (5,)
input_tensor = Input(shape=input_shape)
l1 = Dense(400, activation='softplus')(input_tensor)
l2 = Dense(400, activation='softplus')(l1)
l3 = Dense(400, activation='softplus')(l2)
l4 = Dense(400, activation='softplus')(l3)
output_tensor = Dense(1, activation='softplus')(l4)
model = Model([input_tensor], [output_tensor])

# model.add_loss(K.mean(K.square))

#Implementing custom loss, has to be a nested loss function to customize it 
# def custom_loss(input_tensor, output_tensor):
#     def newloss(y_true, y_pred):
#         mse = K.mean(K.square(y_true - y_pred))
#         grads = lambda output_tensor, input_tensor : K.gradients(output_tensor, input_tensor)
#         u_T = grads(output_tensor, input_tensor)[0][:, 1]
#         u_K = grads(output_tensor, input_tensor)[0][:, 2]
#         gradsK = lambda grads, input_tensor, output_tensor : K.gradients(grads(output_tensor, input_tensor)[0][:, 2], input_tensor)
#         u_KK = gradsK(grads, input_tensor, output_tensor)[0][:, 2]
#         return mse + K.maximum(-1 * u_T, 0) + K.maximum(1 * u_K, 0) + K.maximum(-1 * u_KK, 0)
#     return newloss


#def custom_loss(input_tensor, output_tensor):
def custom_loss(y_true, y_pred):
    def newloss():
        mse = K.mean(K.square(y_true - y_pred))
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            output = model(input_tensor)
        u_T = tape.gradient(output, input_tensor)
        return mse + u_T
    # https://medium.com/analytics-vidhya/tf-gradienttape-explained-for-keras-users-cc3f06276f22
    return newloss

# def newloss(y_true,y_pred):
#     mse = K.mean(K.square(y_true-y_pred))
#     return mse

# compile the model
sgd = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999)
model.compile(loss=custom_loss,
              optimizer='sgd',
              metrics=['mae'])

epochs = 10
batch_size = 100

# I usually only comment out this block whenever I want to load the bs_nn.h5 file, maybe there is something wrong with
# the way I am doing it.
# # fit the model weights.
history = model.fit(x_train_bs, y_train_bs,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test_bs, y_test_bs))
#
#
# model.save('bs_nn.h5')

loaded_model = load_model('bs_nn.h5', custom_objects={'newloss': custom_loss(input_tensor, output_tensor)})
gradsi = loaded_model.optimizer.get_gradients(output_tensor, input_tensor)  # tensor gradients
get_grads = K.function(inputs=input_tensor, outputs=gradsi)
grad_mat_i = get_grads(x_train_bs)

gradsii = loaded_model.optimizer.get_gradients(gradsi, input_tensor)
get_gradsii = K.function(inputs=input_tensor, outputs=gradsii)
grad_mat_ii = get_gradsii(x_train_bs)

error = []
relerror = []

# I believe this may be generating some errors as some of the values are incredibly small
for i in range(len(theta) - 1):
    err = grad_mat_i[0][i, 2] - kderiv[i]
    error.append(err)
    if kderiv[i] != 0:
        relerror.append(err)
    else:
        relerror.append(0)

normerror = [abs((error[i] - np.mean(error))/np.std(error)) for i in range(len(error))]
    
#http://janroman.dhis.org/stud/I2014/BS2/BS_Daniel.pdf

kderiv_pred = grad_mat_i[0][:, 2].flatten()
trendline = np.polyfit(kderiv, kderiv_pred, 1)
coeffs = np.poly1d(trendline)
R = round(np.sum((coeffs(kderiv) - np.mean(kderiv_pred))**2) / np.sum((kderiv_pred - np.mean(kderiv_pred))**2), 4)
Rpara = f"R^2 = {R}"


def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return x, y


x, y = ecdf(error)


perf = plt.figure()

ax1 = perf.add_subplot(121)
# stats.probplot(error, plot = ax3)
ax1.text(0.1, 0.9, s=Rpara, transform=ax1.transAxes)
ax1.scatter(kderiv, kderiv_pred, s=0.5, marker="x")
ax1.set_xlabel('Test Values')
ax1.set_ylabel('Predicted Values')
ax1.plot(theta, coeffs(theta), c='k')
ax1.set_title('Line of Best Fit')

ax2 = perf.add_subplot(122)
ax2.hist(error, bins=50, density=True, color=None, histtype="stepfilled", alpha=0.8)
ax2.set_title('Histogram of Errors')
ax2.set_xlabel('Error')

ax3 = ax2.twinx()
ax3.scatter(x, y, s=0.5, marker='o', c='r')
ax3.set_ylabel('F(x)')
