from scipy.stats import norm
from scipy import stats
import keras
import keras.backend as K
from keras.models import load_model, Model
from keras.layers import Dense, Input
from math import *
import pandas as pd
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt 

df_train = pd.read_csv('/Users/Maximocravero/Desktop/Finance Research 2/European Call NN/training_data_bs.csv', header = None)
df_train = np.array(df_train)
x_train_bs = df_train[:, [0,1,2,3,8]]

input_shape = (5,)
input_tensor = Input(shape = input_shape)
l1 = Dense(400, activation = 'tanh')(input_tensor)
l2 = Dense(400, activation = 'tanh')(l1)
l3 = Dense(400, activation = 'tanh')(l2)
l4 = Dense(400, activation = 'tanh')(l3)
output_tensor = Dense(1, activation = 'relu')(l4)
model = Model(input_tensor, output_tensor)

def custom_loss(input_tensor, output_tensor):
    def newloss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred))
        #T_input = K.gather(input_tensor, 1)
        gradients = K.gradients(output_tensor, input_tensor)[0][:,1]
        #gradients = K.gradients(output_tensor, input_tensor[0])
        return mse + K.maximum(-1*gradients, 0)
    return newloss

loaded_model = load_model('bs_nn.h5', custom_objects = {'newloss' : custom_loss(input_tensor, output_tensor)})

gradients = loaded_model.optimizer.get_gradients(output_tensor, input_tensor) #tensor 

get_gradients = K.function(inputs = input_tensor, outputs = gradients)

gradient_matrix = get_gradients(x_train_bs)