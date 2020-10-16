import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_train = pd.read_csv('/Users/Maximocravero/Desktop/Finance Research 2/European Call NN/training_data_bs.csv', header = None)
df_train = np.array(df_train)
x_train_bs = df_train[:, [0,1,2,3,8]]

df_test = pd.read_csv('/Users/Maximocravero/Desktop/Finance Research 2/European Call NN/test_data_bs.csv', header = None)
df_test = np.array(df_test)
x_test_bs = df_test[:, [0,1,2,3,8]]

y_train_bs = []
y_test_bs = []
theta = []

with open("BS_Constraint/y_train_bs.txt", "r") as f:
  for line in f:
    y_train_bs.append(float(line.strip()))

with open("BS_Constraint/y_test_bs.txt", "r") as f:
  for line in f:
    y_test_bs.append(float(line.strip()))

with open("../EU_Option/BS_Constraint/data/theta.txt", "r") as f:
  for line in f:
    theta.append(float(line.strip()))


#https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data/40995666
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
#Optimization variables
learning_rate = 0.9
epochs = 20
batch_size = 20

#Model parameters
input_shape = 5

#Declaring training data placeholders
x = tf.placeholder(tf.float32, [None, input_shape])
y = tf.placeholder(tf.float32, [None, 1])

w0 = tf.Variable(tf.random_normal([input_shape, 400], stddev = 0.03), name = 'w0')
b0 = tf.Variable(tf.random_normal([400]), name = 'b0')

w1 = tf.Variable(tf.random_normal([400, 400], stddev = 0.03), name = 'w1')
b1 = tf.Variable(tf.random_normal([400]), name = 'b1')

w2 = tf.Variable(tf.random_normal([400, 1], stddev = 0.03), name = 'w2')
b2 = tf.Variable(tf.random_normal([1]), name = 'b2')

hidden_out_1 = tf.add(tf.matmul(x, w0), b0)
hidden_out_1 = tf.nn.relu(hidden_out_1)

hidden_out_2 = tf.add(tf.matmul(hidden_out_1, w1), b1)
hidden_out_2 = tf.nn.relu(hidden_out_2)

y_ = tf.nn.relu(tf.add(tf.matmul(hidden_out_2, w2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 1e4)
loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(y_clipped - y)))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

y_train_bs = np.reshape(y_train_bs, (len(y_train_bs), 1))
y_test_bs = np.reshape(y_test_bs, (len(y_test_bs), 1))
theta = np.reshape(theta, (len(theta), 1))

acc = []
# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(x_train_bs) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = next_batch(batch_size, x_train_bs, y_train_bs)
            batch_y = np.reshape(batch_y, (batch_size, 1))
            _, c = sess.run([optimiser, loss], 
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
            #print(avg_cost)
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: x_test_bs, y: y_test_bs}))



