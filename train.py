import wandb
import argparse
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from neuralNetwork import FeedForwardNN

#taking inputs from the command line with format given in the assignment
def input():
  args = argparse.ArgumentParser(description='CS6910-Assignment 1:')
  args.add_argument('-wp','--wandb_project', type=str, default = "CS6910-Assignment 1")
  args.add_argument('-we','--wandb_entity', type=str, default = "cs22m048")
  args.add_argument('-d','--dataset', type=str, default = "fashion_mnist")
  args.add_argument('-e','--epochs', type=int, default = 15)
  args.add_argument('-b','--batch_size', type=int, default = 32)
  args.add_argument('-l','--loss', type=str, default = "cross_entropy")
  args.add_argument('-o','--optimizer', type=str, default = "adam")
  args.add_argument('-lr','--learning_rate',type=float, default = 0.0001)
  args.add_argument('-m','--momentum',type=float, default = 0.8)
  args.add_argument('-beta','--beta',type=float, default = 0.7)
  args.add_argument('-beta1','--beta1',type=float, default = 0.9)
  args.add_argument('-beta2','--beta2',type=float, default = 0.999)
  args.add_argument('-eps','--epsilon',type=float, default = 0.000001)
  args.add_argument('-w_d','--weight_decay',type=float, default =0.0005)
  args.add_argument('-w_i','--weight_init', type=str, default = "Xavier")
  args.add_argument('-nhl','--num_layers', type=int, default = 2)
  args.add_argument('-sz','--hidden_size', type=int, default = 64)
  args.add_argument('-a','--activation', type=str, default = "ReLU")

  arguments = args.parse_args()

  return arguments

arguments=input()

if(arguments.dataset=="fashion_mnist"):
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
if(arguments.dataset=="mnist"):
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

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

wandb.login()
sweep_config = {
  "name": "test Sweep",
  "method": "random",
  "metric":{
    "name": "accuracytest",
    "goal": "maximize"
  },
  "parameters": {
        "epochs": {
            "values": [arguments.epochs]
        },

        "initialize": {
            "values": [arguments.weight_init]
        },

        "noOfHL": {
            "values": [arguments.num_layers]
        },
        
        "NeuronsPL": {
            "values": [arguments.hidden_size]
        },

        "activationFunc": {
            "values": [arguments.activation]
        },
        
        "learningRate": {
            "values": [arguments.learning_rate]
        },
        
        "weightDecay": {
            "values": [arguments.weight_decay]
        },

        "optimizer": {
            "values": [arguments.optimizer]
        },
                    
        "batchSize": {
            "values": [arguments.batch_size]
        },
        "lossfunction":{
            "values":[arguments.loss]
        }
        
        
    }
}
sweep_id = wandb.sweep(sweep_config,project='train.py', entity='cs22m048')
def train(config=None):
    wandb.init(config=config)
    wandb.run.name = "HL-" + str(wandb.config.noOfHL) + "Neuron-" + str(wandb.config.NeuronsPL) + "Opt-" + wandb.config.optimizer + "Act-" + wandb.config.activationFunc + "LR-" + str(wandb.config.learningRate) +"WD-" + str(wandb.config.weightDecay) + "BS-"+str(wandb.config.batchSize) + "Init-" + wandb.config.initialize

    CONFIG = wandb.config

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
        gamma=arguments.momentum,
        Beta=arguments.beta,
        Beta1=arguments.beta1,
        Beta2=arguments.beta2,
        epsilon=arguments.epsilon
    )
    loss,accuracytrain,accuracytest=FWNN.optimizer()
    wandb.log({'accuracytest':accuracytest});
wandb.agent(sweep_id,train,count=1);