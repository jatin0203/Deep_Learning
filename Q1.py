import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from neuralNetwork import FeedForwardNN

#importing the dataset from keras library
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_name = [
        "T-shirt/Top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    ]
#one data image per label
Images= []
Image_name= []
fig,ax=plt.subplots(nrows=5,ncols=5,figsize=(15,15))
for i in range(10):
    if i<5:
        ax[0][i].set_title(class_name[i])
        ax[0][i].axis("off")
        x=x_train[y_train==i]
        ax[0][i].imshow(x[0],cmap="gray")
        Images.append(x[0])
        Image_name.append(class_name[i])
    else:
        ax[1][i-5].set_title(class_name[i])
        ax[1][i-5].axis("off")
        x=x_train[y_train==i]
        ax[1][i-5].imshow(x[0],cmap="gray")
        Images.append(x[0])
        Image_name.append(class_name[i])

wandb.init(project='Q1', entity='cs22m048')
wandb.log({"Image": [wandb.Image(Image,caption=name) for Image,name in zip(Images,Image_name)]})
