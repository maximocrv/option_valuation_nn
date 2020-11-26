import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
from tensorflow.keras.losses import MeanSquaredError


def save_model(model, model_path: str):
    caller = np.random.uniform(0, 2, (1, 4)).astype(np.float32)
    model(caller)
    model.save(model_path)


def load_model(model_path: str):
    model = tf.keras.models.load_model(model_path)
    model.compile()
    return model


class CustomMSE(MeanSquaredError):
    def __init__(self, model, x_batch, reduction=tf.keras.losses.Reduction.AUTO, name='custom_mse'):
        super().__init__(reduction=reduction, name=name)
        self.model = model
        self.x_batch = x_batch

    def __call__(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.math.square(y_true-y_pred))
        with tf.GradientTape() as tape:
            logits = self.model.call(self.x_batch)
            u_k = tf.reduce_mean(tf.maximum(0, tape.gradient(logits, self.x_batch)[:, 2]))
        return mse + u_k


class MyModel(tf.keras.Model):
    def __init__(self, input_dim, mode, activation='relu', sc_weights=(1, 1), epsilon=0):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        self.mode = mode
        self.sc_weights = sc_weights
        self.epsilon = epsilon
        self.dense1 = Dense(400, activation=activation, kernel_initializer=initializers.glorot_uniform(),
                            input_dim=self.input_dim)
        self.dense2 = Dense(400, activation=activation, kernel_initializer=initializers.glorot_uniform())
        self.dense3 = Dense(400, activation=activation, kernel_initializer=initializers.glorot_uniform())
        self.dense4 = Dense(400, activation=activation, kernel_initializer=initializers.glorot_uniform())
        self.dense_out = Dense(1, activation=activation, kernel_initializer=initializers.glorot_uniform())

    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32, name='inputs')])   #CHECK tf.saved_model.save docs!
    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense_out(x)

    def get_loss(self, x, y):
        with tf.GradientTape() as l_tape:
            l_tape.watch(x)
            y_pred = self.call(x)
        grad_mat = l_tape.gradient(y_pred, x)

        if self.mode == 'mse':
            return tf.reduce_mean(tf.math.square(y_pred - y[:, tf.newaxis]))

        # WATCH OUT WITH GRAD_MAT HERE IF YOU ARE ADJUSTING THE INPUTS!!!
        elif self.mode == 'u_T':
            return tf.reduce_mean(tf.math.square(y_pred - y[:, tf.newaxis])) \
                   + self.sc_weights[0] * tf.reduce_mean(tf.maximum(0, -1 * (grad_mat[:, 0]) - self.epsilon))
            # threshold to force strict inequality
            # + 5e-3 * tf.reduce_mean(x[:, 0] * tf.maximum(0, -1 * grad_mat[:, 0]))

        elif self.mode == 'u_K':
            return tf.reduce_mean(tf.math.square(y_pred - y[:, tf.newaxis])) \
                   + self.sc_weights[1] * tf.reduce_mean(tf.maximum(0, 1 * grad_mat[:, 1]))

        elif self.mode == 'u_KT':
            return tf.reduce_mean(tf.math.square(y_pred - y[:, tf.newaxis])) \
                   + self.sc_weights[0] * tf.reduce_mean(tf.maximum(0, -1 * (grad_mat[:, 0]) - self.epsilon)) \
                   + self.sc_weights[1] * tf.reduce_mean(tf.maximum(0, 1 * grad_mat[:, 1]))
        else:
            return 'Error, please enter a valid loss mode'

    def get_grad_and_loss(self, x, y):
        with tf.GradientTape() as gl_tape:
            gl_tape.watch(tf.convert_to_tensor(x))
            loss = self.get_loss(x, y)
        g = gl_tape.gradient(loss, self.trainable_weights)
        return g, loss
