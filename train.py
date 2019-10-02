from model import Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#Normalize the data
x_train = x_train / 255
x_train = (x_train>0.5)*1

x_test  = x_test  / 255
x_test  = (x_test>0.5)*1

#Init the model
m = Model(name = 'test')

#Train the model and save the network
m.train(np.expand_dims(x_train,-1), y_train, val_data = (np.expand_dims(x_test,-1), y_test), epochs = 15)
m.save_summary()
m.save_model()
