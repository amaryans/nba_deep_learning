### Main Neural Network For the 3 point shot predictor

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from dataloader import *

### All of this is from old code. The bigges change will be in "model.py" as it is how the model is built

#The folder where your dataset is. Note that is must end with a '/'
direc = 'data/'

plot = False                     #Set True if you wish plots and visualizations

"""Hyperparameters"""
config = {}
config['MDN'] = MDN = False      #Set to falso for only the classification network
config['num_layers'] = 2         #Number of layers for the LSTM
config['hidden_size'] = 64       #Hidden size of the LSTM
config['max_grad_norm'] = 1      #Clip the gradients during training
config['batch_size'] = batch_size = 64
config['sl'] = sl = 12           #Sequence length to extract data
config['mixtures'] = 3           #Number of mixtures for the MDN
config['learning_rate'] = .005   #Initial learning rate


ratio = 0.8                      #Ratio for train-val split
plot_every = 100                 #How often do you want terminal output for the performances
max_iterations = 20000           #Maximum number of training iterations
dropout = 0.7                    #Dropout rate in the fully connected layer

db = 5                           #distance to basket to stop trajectories

"""Load the data"""
#The name of the dataset. Note that it must end with '.csv'
csv_file = 'seq_all.csv'
#Load an instance
center = np.array([5.25, 25.0, 10.0])   #Center of the basket for the dataset
dl = DataLoad(direc,csv_file,center)

#Munge the data. Arguments see the class
dl.munge_data(11,sl,db, False)


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

# The four coordinates of the shape are x, y, z, and time to the shot.
N,crd,_ = X_train.shape
print(N,crd,_)
Nval = X_val.shape[0]

config['crd'] = crd            #Number of coordinates. usually three (X,Y,Z) and time (game_clock)



# Batch size is number of samples (e.g. number of shots), crd is the number of features, sl is the number of time steps
rnn_input = layers.Input(shape = (crd, sl))
cnn_input = layers.Input(shape = (50, 94, 3))
cnn = layers.Conv2D(20, kernel_size=(5, 5), activation = 'relu')(cnn_input)
cnn = layers.Conv2D(5, kernel_size = (5, 5), padding = 'valid')(cnn)
cnn = layers.Flatten()(cnn)
cnn = layers.Dense(crd* sl)(cnn)
cnn = layers.Reshape((crd, sl))(cnn)
#rnn_input = layers.Concatenate()([cnn, rnn_input])
rnn = layers.Conv1D(12, 1, padding = 'same')(rnn_input)
rnn = layers.Concatenate()([cnn, rnn])
rnn = layers.LSTM(10)(rnn)
rnn = layers.Dense(10)(rnn)
model_output = layers.Dense(1)(rnn)

model = keras.Model(inputs=[rnn_input, cnn_input], outputs=model_output)
model.summary()


# Default Hidden Layers which will be in each of the branches
# encoder_input = keras.Input(shape=(32, 32, 1))
# x = Conv2D(60, kernel_size=(5,5), strides=1, activation="relu")(encoder_input)
# x = MaxPooling2D(2)(x)
# x = Conv2D(40, kernel_size=(3,3), strides=1, activation='relu')(x)
# x = Dropout(dropout_val)(x)
# x = Flatten()(x)
# x = Dense(140)(x)
# x = Activation('relu')(x)