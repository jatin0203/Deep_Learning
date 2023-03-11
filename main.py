import wandb
import numpy as np
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
x_valid = x_train[:num_valid,:] 
y_valid = y_train[:num_valid] 

x_train = x_train[num_valid:, :] 
y_train = y_train[num_valid:]  

#defining sweep configuration
wandb.login()
sweep_config = {
  "name": "bayes Sweep",
  "method": "bayes",
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
            "values": ['RELU','SIGMOID','TANH']
        },
        
        "learningRate": {
            "values": [0.1, 0.01, 0.001,0.0001]
        },
        
        "weightDecay": {
            "values": [0,0.5,0.0005]
        },

        "optimizer": {
            "values": ["SGD", "MGD", "NAG", "RMSPROP", "ADAM","NADAM"]
        },
                    
        "batchSize": {
            "values": [32, 64, 128]
        },
        "lossfunction":{
            "values":['CROSS']
        }
        
        
    }
}

sweep_id = wandb.sweep(sweep_config,project='testing1', entity='cs22m048')

def train(config=None):
    config_defaults=dict(
        NeuronsPL=32,
        epochs=10,
        noOfHL=2,
        lossfunction="CROSS",
        activationFunc="SIGMOID",
        learningRate=0.001,
        weightDecay=0.0005,
        batchSize=32,
        optimizer="NADAM",
        gamma=0.8,
        initialize="XAVIER",
        Beta=0.7,
        Beta1=0.9,
        Beta2=0.999,
        epsilon=0.00001
    )
    
    wandb.init(config = config)
        
    wandb.run.name = "HL-" + str(wandb.config.noOfHL) + "Neuron-" + str(wandb.config.NeuronsPL) + "Opt-" + wandb.config.optimizer + "Act-" + wandb.config.activationFunc + "LR-" + str(wandb.config.learningRate) +"WD-" + str(wandb.config.weightDecay) + "BS-"+str(wandb.config.batchSize) + "Init-" + wandb.config.initialize + "Ep-"+ str(wandb.config.epochs) 
    CONFIG = wandb.config
    
    #creating the object
    FWNN=FeedForwardNN(
        epochs=CONFIG.epochs,
        noOfHL=CONFIG.noOfHL,
        NeuronsPL=CONFIG.NeuronsPL,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        x_test=x_test,
        y_test=y_test,
        optimizer=CONFIG.optimizer,
        activationfunction=CONFIG.activationFunc,
        learningRate=CONFIG.learningRate,
        weightDecay=CONFIG.weightDecay,
        batchSize=CONFIG.batchSize,
        initialize=CONFIG.initialize,
        lossfunction=CONFIG.lossfunction,
        gamma=0.9,
        Beta=0.5,
        Beta1=0.9,
        Beta2=0.999,
        epsilon=0.0001
    )
    
    
    #training the model using sgd
    loss,accuracytrain,accuracytest=FWNN.optimizer()
    

wandb.agent(sweep_id, train, count = 50)

