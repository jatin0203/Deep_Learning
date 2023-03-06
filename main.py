#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import wandb

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.datasets import fashion_mnist
from neuralNetwork import FeedForwardNN

#importing the dataset from keras library
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# #one data image per label
# fig,ax=plt.subplots(nrows=10,figsize=(15,15))
# for i in range(10):
#     ax[i].set_title("\n class {} image".format(i))
#     ax[i].axis("off")
#     x=x_train[y_train==i]
#     ax[i].imshow(x[0,:,:],cmap="gray")

#normalizing the data between 0-1
x_train=x_train/255
x_train=x_train.astype(float);
x_test=x_test/255
x_test=x_test.astype(float);

#flattening the data points to 1D
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)

#creating validation set
num_valid = int(0.1 * x_train.shape[0])  #using 10% of the data as validation test for the model and remaining for train
x_valid = x_train[:num_valid, :] 
y_valid = y_train[:num_valid] 

x_train = x_train[num_valid:, :] 
y_train = y_train[num_valid:]  

#defining sweep configuration
sweep_config = {
  "name": "Random Sweep",
  "method": "random",
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  },
  "parameters": {
        "epochs": {
            "values": [5, 10, 15]
        },

        "initialize": {
            "values": ["RANDOM", "XAVIER"]
        },

        "noOfHL": {
            "values": [2, 3, 4]
        },
        
        
        "NeuronsPL": {
            "values": [32, 64, 128]
        },
        
        "activationFunc": {
            "values": ['RELU', 'SIGMOID', 'TANH', 'SINH']
        },
        
        "learningRate": {
            "values": [0.1, 0.01, 0.001, 0.0001, 0.00001]
        },
        
        "optimizer": {
            "values": ["SGD", "MGD", "NAG", "RMSPROP", "ADAM","NADAM"]
        },
                    
        "batchSize": {
            "values": [16, 32, 64, 128]
        }
        
        
    }
}

sweep_id = wandb.sweep(sweep_config,project='', entity='')

def train():
    config_defaults=dict(
        NeuronsPL=32,
        epochs=10,
        noOfHL=2,
        noofClass=10,
        lossfunction="CROSS",
        activationFunc="SIGMOID",
        learningRate=0.001,
        batchSize=32,
        optimizer="NADAM",
        gamma=0.8,
        initialize="XAVIER",
        Beta=0.7,
        Beta1=0.9,
        Beta2=0.999,
        epsilon=0.00001
    )
    
    wandb.init(config = config_defaults)
        
    wandb.run.name = "hl_" + str(wandb.config.noOfHL) + "_hn_" + str(wandb.config.num_hidden_neurons) + "_opt_" + wandb.config.optimizer + "_act_" + wandb.config.activation + "_lr_" + str(wandb.config.learning_rate) + "_bs_"+str(wandb.config.batch_size) + "_init_" + wandb.config.initializer + "_ep_"+ str(wandb.config.max_epochs)+ "_l2_" + str(wandb.config.weight_decay) 
    CONFIG = wandb.config
    
    #creating the object
    FWNN=FeedForwardNN(
        epochs=CONFIG.epochs,
        noOfHL=CONFIG.noOfHL,
        NeuronsPL=CONFIG.NeuronsPL,
        noofClass=CONFIG.noofClass,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
        optimizer=CONFIG.optimizer,
        activationFunc=CONFIG.activationFunc,
        learningRate=CONFIG.learningRate,
        batchSize=CONFIG.batchSize,
        initialize=CONFIG.initialize,
        lossfunction=CONFIG.lossfunction,
        gamma=CONFIG.gamma,
        Beta=CONFIG.Beta,
        Beta1=CONFIG.Beta1,
        Beta2=CONFIG.Beta2,
        epsilon=CONFIG.epsilon
    )
    
    #predicting before training the model
    y_pred=FWNN.calculatePredClasses("train")
    print(y_pred)
    
    #training the model using sgd
    loss,accuracytrain,accuracytest=FWNN.optimizer()


# In[ ]:




