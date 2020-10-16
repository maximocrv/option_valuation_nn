#  heston_nn is a keras Model
import numpy as np
import keras.backend as K
from keras.models import model_from_json
from keras import optimizers
import pandas as pd

json_file = open('EU_Call.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('EU_Call_model.h5')

weights = loaded_model.weights  # weight tensors
gradients = loaded_model.optimizer.get_gradients(loaded_model.total_loss, weights)  # gradient tensors

input_tensors = [loaded_model.inputs,
                 loaded_model.sample_weights[0],  # sample weights
                 loaded_model.targets[0],  # labels
                 K.learning_phase()]  # train or test mode]

get_gradients = K.function(inputs = input_tensors, outputs = gradients)

df_train = pd.read_excel(r'C:\Users\maxic\Desktop\Finance Research\European Call NN\training_data.xlsx', header = None)
xtrain = df_train[df_train.columns[0:8]]


sample_lens = len(xtrain[0])
print('the number of samples is ', sample_lens)
inputs = [np.reshape(xtrain[0], (sample_lens, 1)),  # X input data, including x0=Stock, x1=Time, .....
          np.reshape(xtrain[1], (sample_lens, 1)),
          np.reshape(xtrain[2], (sample_lens, 1)),
          np.reshape(xtrain[3], (sample_lens, 1)),
          np.reshape(xtrain[4], (sample_lens, 1)),
          np.reshape(xtrain[5], (sample_lens, 1)),
          np.reshape(xtrain[6], (sample_lens, 1)),
          np.reshape(xtrain[7], (sample_lens, 1)),
          np.ones((sample_lens,)),  # sample weights
          ytrue,  # y labels
          0  # learning phase in TEST mode
          ]

gradient_matrix = get_gradients(inputs)