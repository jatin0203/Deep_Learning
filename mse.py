import wandb
from keras.datasets import fashion_mnist
from neuralNetwork import FeedForwardNN
from optimizer import Optimizers

#importing the dataset from keras library
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
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
  "name": "Random Sweep",
  "method": "bayes",
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  },
  "parameters": {
        "epochs": {
            "values": [15]
        },

        "initialize": {
            "values": ["Xavier"]
        },

        "noOfHL": {
            "values": [2, 4]
        },
        
        
        "NeuronsPL": {
            "values": [32, 64]
        },
        
        "activationFunc": {
            "values": ['ReLU','sigmoid']
        },
        
        "learningRate": {
            "values": [0.01, 0.0001]
        },

        "weightDecay": {
            "values": [0.0005]
        },
        
        "optimizer": {
            "values": ["adam","nadam"]
        },
                    
        "batchSize": {
            "values": [64, 128]
        },
        "lossfunction":{
            "values":['mean_squared_error','cross_entropy']
        }
        
        
    }
}

sweep_id = wandb.sweep(sweep_config,project='msetest', entity='cs22m048')

def train(config=None):
    wandb.init(config = config)
        
    wandb.run.name = "HL-" + str(wandb.config.noOfHL) + "Neuron-" + str(wandb.config.NeuronsPL) + "Opt-" + wandb.config.optimizer + "Act-" + wandb.config.activationFunc + "LR-" + str(wandb.config.learningRate) + "BS-"+str(wandb.config.batchSize) + "Init-" + wandb.config.initialize + "Ep-"+ str(wandb.config.epochs) 
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
    Opt=Optimizers()
    loss,accuracytrain,accuracytest=Opt.optimize(FWNN)
    

wandb.agent(sweep_id, train, count = 60)

