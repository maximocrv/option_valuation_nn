from math import *

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from scipy.stats import norm
from operator import itemgetter
from sklearn.model_selection import train_test_split


def BS_Call(xarray: np.array) -> tuple:  # Calculating BS option values and Greeks
    S, T, K, r, v, = itemgetter(0, 1, 2, 3, 4)(xarray)

    T_sqrt = sqrt(T)

    d1 = (log(float(S)/K)+(r+v*v/2.)*T)/(v*T_sqrt)
    d2 = d1-v*T_sqrt

    dCdK = -exp(-r*T)*norm.cdf(d2)
    dPdK = exp(-r*T)*norm.cdf(-d2)
    dCdT = -1*(-(S*v*norm.pdf(d1))/(2*T_sqrt)-r*K*exp(-r*T)*norm.cdf(d2))  # *-1 due to dV/dt = -dV/dT!! Q10.5 Higham
    # dPdT = -1*(-(S*v*norm.pdf(d1))/(2*T_sqrt) + r*K*exp(-r*T)*norm.cdf(-d2))  # Higham 10.7 for Greeks Put-Call Parity
    dPdT = dCdT - K*r*exp(-r*T)
    return dCdK, dPdK, dCdT, dPdT


def gen_training_data(x: np.array):
    dCdK = []
    dPdK = []
    dCdT = []
    dPdT = []

    for i in range(len(x)):
        ck, pk, ct, pt = BS_Call(x[i, :])
        dCdK.append(ck)
        dPdK.append(pk)
        dCdT.append(ct)
        dPdT.append(pt)

    with open('EU_Option/BS_Constraint/data/dCdK.txt', 'w') as f:
        for item in dCdK:
            f.write(str(item) + "\n")

    with open('EU_Option/BS_Constraint/data/dPdK.txt', 'w') as f:
        for item in dPdK:
            f.write(str(item) + "\n")

    with open('EU_Option/BS_Constraint/data/dCdT.txt', 'w') as f:
        for item in dCdT:
            f.write(str(item) + "\n")

    with open('EU_Option/BS_Constraint/data/dPdT.txt', 'w') as f:
        for item in dPdT:
            f.write(str(item) + "\n")


def load_training_data():
    dCdK = []
    dPdK = []
    dCdT = []
    dPdT = []

    with open("EU_Option/BS_Constraint/data/dCdK.txt", "r") as f:
        for line in f:
            dCdK.append(float(line.strip()))

    with open("EU_Option/BS_Constraint/data/dPdK.txt", "r") as f:
        for line in f:
            dPdK.append(float(line.strip()))

    with open("EU_Option/BS_Constraint/data/dCdT.txt", "r") as f:
        for line in f:
            dCdT.append(float(line.strip()))

    with open("EU_Option/BS_Constraint/data/dPdT.txt", "r") as f:
        for line in f:
            dPdT.append(float(line.strip()))

    return np.array(dCdK), np.array(dPdK), np.array(dCdT), np.array(dPdT)


def gen_model_data(x: np.array, y: np.array, random_state=1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=random_state)

    return x_train, y_train, x_val, y_val, x_test, y_test


def gen_tf_model_data(x_train, y_train, x_val, y_val, x_test, y_test, batch):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch)

    return train_dataset, val_dataset, test_dataset