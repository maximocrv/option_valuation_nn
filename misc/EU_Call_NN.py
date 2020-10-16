import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Reshape, Input
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K
from keras import initializers
from keras import optimizers
import pandas as pd
import tensorflow as tf

#tf.enable_eager_execution()

#df_train = pd.read_excel(r'C:\Users\maxic\Desktop\Finance Research\European Call NN\training_data.xlsx', header = None)
df_train = pd.read_excel('/Users/Maximocravero/Desktop/Finance Research 2/European Call NN/training_data.xlsx', header = None)
df_train = np.array(df_train)
#df_test = pd.read_excel(r'C:\Users\maxic\Desktop\Finance Research\European Call NN\test_data.xlsx', header = None)
df_test = pd.read_excel('/Users/Maximocravero/Desktop/Finance Research 2/European Call NN/test_data.xlsx', header = None)
df_test = np.array(df_test)

x_train = df_train[:,0:8]
y_train = df_train[:,-1]
y_train = y_train.reshape(len(y_train),-1)
x_test = df_test[:,0:8]
y_test = df_train[:,-1]
y_test = y_test.reshape(len(y_test), -1)


input_shape = (8,)

# model = Sequential()
# initializers.glorot_uniform(seed = None)
# model.add(Dense(400, activation = 'relu', input_shape = input_shape))
# model.add(Dense(400, activation = 'relu'))
# model.add(Dense(400, activation = 'relu'))
# model.add(Dense(400, activation = 'relu'))
# model.add(Dense(1, activation = 'relu'))

input_tensor = Input(shape = input_shape)
l1 = Dense(400, activation = 'relu')(input_tensor)
l2 = Dense(400, activation = 'relu')(l1)
l3 = Dense(400, activation = 'relu')(l2)
l4 = Dense(400, activation = 'relu')(l3)
output_tensor = Dense(1, activation = 'relu')(l4)
model = Model(input_tensor, output_tensor)

#file_heston = 'EU_Call_model.h5'
#loaded_model = load_model(file_heston, custom_objects={'newloss': newloss(y,yhat)})
#loaded_model = load_model(file_heston)


# input_tensors = [*loaded_model.inputs,
#                   loaded_model.sample_weights[0],  # sample weights
#                   loaded_model.targets[0],  # labels
#                   K.learning_phase()]  # train or test mode]

# weights = loaded_model.weights  # weight tensors
# gradients = loaded_model.optimizer.get_gradients(loaded_model.output, loaded_model.input)  # gradient tensors


# get_gradients = K.function(inputs = input_tensors, outputs = gradients)

# inputs = [x_train,
#           np.ones(x_train.shape[0]),
#           y_train,  # y labels
#           0]  # learning phase in TEST mode

# gradient_matrix = get_gradients(inputs)

# def newloss(y,yhat):
#     msee = K.mean(K.square(y-yhat))
#     #grad = max(-get_gradients(loaded_model.input), 0) #unsure how to actually incorporate the gradient here when defining the loss function
#     grad = K.max(-tf.gradients(loaded_model.output, loaded_model.input)[0], 0)
#     return msee + grad
#     #return msee

def custom_loss(input_tensor, output_tensor):
    def newloss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred))
        gradients = K.gradients(output_tensor, input_tensor)[0][:,1]
        #gradients = K.gradients(output_tensor, input_tensor[1])
        return mse + K.maximum(-1*gradients, 0)
    return newloss


sgd = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(loss = custom_loss(input_tensor, output_tensor),
              optimizer = 'sgd',
              metrics = ['mae'])

#epochs = 200
#batch_size = 1024
epochs = 15
batch_size = 50
# Fit the model weights.
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save("EU_Call_model.h5")


# y_predicted = loaded_model.predict(x_test)
# diff = y_test - y_predicted.flatten()
#
# trendline = np.polyfit(y_test, y_predicted.flatten(), 1)
# coeffs = np.poly1d(trendline)
# R = round(np.sum((coeffs(y_test) - np.mean(y_predicted.flatten()))**2) / np.sum((y_predicted - np.mean(y_predicted.flatten()))**2), 4)
# Rpara = "R^2 = {}".format(R)
#
# def ecdf(data):
#     """ Compute ECDF """
#     x = np.sort(data)
#     n = x.size
#     y = np.arange(1, n+1) / n
#     return(x,y)
#
# x, y = ecdf(diff)
#
# perf = plt.figure()
#
# ax1 = perf.add_subplot(121)
# ax1.text(0.1, 0.9, s = Rpara, transform = ax1.transAxes)
# ax1.scatter(y_test, y_predicted, s = 0.5, marker = "x")
# ax1.set_xlabel('Test Values')
# ax1.set_ylabel('Predicted Values')
# ax1.plot(y_test, coeffs(y_test), c = 'k')
# ax1.set_title('Line of Best Fit')
#
# ax2 = perf.add_subplot(122)
# ax2.hist(diff, bins = 50, density = True, color = None, histtype = "stepfilled", alpha = 0.8)
# ax2.set_title('Histogram of Errors')
# ax2.set_xlabel('Error')
#
# ax3 = ax2.twinx()
# ax3.scatter(x,y, s=0.5, marker = 'o', c = 'r')
# ax3.set_ylabel('F(x)')




## df_train = pd.read_excel(r'C:\Users\maxic\Desktop\Finance Research\European Call NN\training_data.xlsx', header = None)
## df_test = pd.read_excel(r'C:\Users\maxic\Desktop\Finance Research\European Call NN\test_data.xlsx', header = None)
##
## x_train = df_train[df_train.columns[0:8]]
## y_train = df_train[df_train.columns[8]]
##
## x_test = df_test[df_test.columns[0:8]]
## y_test = df_test[df_test.columns[8]]


# #saving model
# EU_Call_model_json = model.to_json()
# with open("EU_Call.json", "w") as json_file:
#     json_file.write(EU_Call_model_json)


#loading model
## json_file = open('EU_Call.json', 'r')
## loaded_model_json = json_file.read()
## json_file.close()
## loaded_model = model_from_json(loaded_model_json)
## loaded_model.load_weights('EU_Call_model.h5')

#https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
#https://stackoverflow.com/questions/49688134/tensorflow-session-inside-keras-custom-loss-function