import numpy as np
import tensorflow as tf
import csv
import os
import time


class Model():

    def __init__(self, name=None, model_path = None):
        '''
        -Initializes a model class with a given name.
        -Creates directory for saving model, its weights, summary and results.
        -Initializes optimizer(Adam) and compiles the model
        '''

        #Menage model's name
        if name is None:
            self.NAME = "NotName-{}".format(int(time.time())%1000)
        else:
            self.NAME = name
        print('Model name:', self.NAME)

        #Menage model's dir
        if os.path.isdir(self.NAME):
            print('Model\'s dir already exists')
        else:
            os.mkdir(self.NAME)

        #Define or load the model
        if model_path is None:
            self.model = self.define_network()
            self.optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
            self.model.compile(optimizer=self.optimizer,
                               loss='sparse_categorical_crossentropy',
                               metrics = ['accuracy'])
        else:
            self.model = self.load_model(model_path)

    def define_network(self, input_shape = (28,28,1)):
        '''
        Uses keras to create a neural network
        '''

        #Input
        main_input = tf.keras.Input(shape = input_shape, name = 'main_input')

        #Network Body
        x = tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation = 'relu')(main_input)
        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units = 128, activation = 'relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        #Output
        output = tf.keras.layers.Dense(units = 10, activation = 'softmax')(x)

        #Model
        model = tf.keras.models.Model(inputs = [main_input], outputs = output)

        return model

    def train(self, data, labels, epochs = 5, batch_size = None, val_data=None):
        '''
        Trains the model
        '''
        self.model.fit(data, labels, epochs = epochs, batch_size=batch_size, validation_data = val_data)

    def predict(self, data):
        '''
        Predicts the output for data
        '''

        return self.model.predict(data)

    def save_summary(self):
        '''
        Saves summary in a model's dir
        '''
        with open(self.NAME+"/summary.txt","w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

    def save_model(self):
        '''
		Saves the model
		'''
        tf.keras.models.save_model(self.model, os.path.join(self.NAME,'model.h5'))

    def load_model(self,filename):
        '''
		Loads the model
		'''
        return tf.keras.models.load_model(filename)
