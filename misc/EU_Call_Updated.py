import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, load_model, Model
from keras.models import model_from_json
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
#df_train = pd.read_excel(r'.\training_data.xlsx', header = None)
df_train = np.array(df_train)
#df_test = pd.read_excel(r'C:\Users\maxic\Desktop\Finance Research\European Call NN\test_data.xlsx', header = None)
df_test = pd.read_excel('/Users/Maximocravero/Desktop/Finance Research 2/European Call NN/test_data.xlsx', header = None)
#df_test = pd.read_excel(r'.\test_data.xlsx', header = None)
df_test = np.array(df_test)


x_train = df_train[:,0:8]
y_train = df_train[:,-1]
y_train = y_train.reshape(len(y_train),-1)
x_test = df_test[:,0:8]
y_test = df_train[:,-1]
y_test = y_test.reshape(len(y_test), -1)


input_shape = (8,)
input_tensor = Input(shape = input_shape)
l1 = Dense(400, activation = 'relu')(input_tensor)
l2 = Dense(400, activation = 'relu')(l1)
l3 = Dense(400, activation = 'relu')(l2)
l4 = Dense(400, activation = 'relu')(l3)
output_tensor = Dense(1, activation = 'relu')(l4)
model = Model(input_tensor, output_tensor)

def custom_loss(input_tensor, output_tensor):
    def newloss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred))
        gradients = K.gradients(output_tensor, input_tensor)[0]
        return mse + K.maximum(-1*gradients, 0)
    return newloss
        
sgd = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(loss = custom_loss(input_tensor, output_tensor),
              optimizer = 'sgd',
              metrics = ['mae'])


epochs = 10
batch_size = 100
# Fit the model weights.
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save('heston_nn.h5')

#from keras.models import load_model

# serialize model to JSON
#  the keras model which is trained is defined as 'model' in this example
#model_json = model.to_json()


# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")

 
# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")

# loaded_model.compile(loss='custom_loss', optimizer='rmsprop', metrics=['accuracy'])

