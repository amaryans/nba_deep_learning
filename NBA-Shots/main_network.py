### Main Neural Network For the 3 point shot predictor

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from dataloader import *

class RecurrentMode(Model):
    def __init__(self, num_timesteps, *args, **kwargs):
        self.num_timesteps = num_timesteps
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        inputs = layers.Input((None, None, input_shape[-1]))
        x = layers.Conv2D(64, (3, 3), activation='relu'))(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(3, activation='linear')(x)
        self.model = Model(inputs=[inputs], outputs=[x])

    def call(self, inputs, **kwargs):
        x = inputs
        for i in range(self.num_timestaps):
            x = self.model(x)
        return x
### All of this is from old code. The bigges change will be in "model.py" as it is how the model is built

"""Hyperparameters"""
config = {}
config['MDN'] = MDN = False       #Set to falso for only the classification network
config['num_layers'] = 2         #Number of layers for the LSTM
config['hidden_size'] = 64     #Hidden size of the LSTM
config['max_grad_norm'] = 1      #Clip the gradients during training
config['batch_size'] = batch_size = 64
config['sl'] = sl = 12           #Sequence length to extract data
config['mixtures'] = 3           #Number of mixtures for the MDN
config['learning_rate'] = .005   #Initial learning rate


ratio = 0.8                      #Ratio for train-val split
plot_every = 100                 #How often do you want terminal output for the performances
max_iterations = 20000             #Maximum number of training iterations
dropout = 0.7                    #Dropout rate in the fully connected layer

db = 5                            #distance to basket to stop trajectories



"""Load the data"""
#The name of the dataset. Note that it must end with '.csv'
csv_file = 'seq_all.csv'
#Load an instance
center = np.array([5.25, 25.0, 10.0])   #Center of the basket for the dataset
dl = DataLoad(direc,csv_file, center)

#Munge the data. Arguments see the class
dl.munge_data(11,sl,db,True)


#Center the data
dl.center_data(center)
dl.split_train_test(ratio = 0.8)
data_dict = dl.data
plot = 0
if plot:
  dl.plot_traj_2d(20,'at %.0f feet from basket'%db)

X_train = np.transpose(data_dict['X_train'],[0,2,1])
y_train = data_dict['y_train']
X_val = np.transpose(data_dict['X_val'],[0,2,1])
y_val = data_dict['y_val']

N,crd,_ = X_train.shape
Nval = X_val.shape[0]

config['crd'] = crd            #Number of coordinates. usually three (X,Y,Z) and time (game_clock)

#How many epochs ill we train
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

model = Model(config)