# -*- coding: utf-8 -*-
"""
MNIST Digit 

@author: Garrett
"""
from __future__ import print_function
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
#from keras.optimizers import Adam, RMSProp
from keras import backend as K

import matplotlib.pyplot as plt
#%matplotlib inline

batch_size=120
num_classes=10
img_rows, img_cols = 28, 28

#Load Train/Test
train = pd.read_csv("C:\Users\Garrett\Documents\Projects\MNIST_Kaggle\MNIST_digits\data\\train.csv")
print(train.shape)
train.head()

test = pd.read_csv("C:\Users\Garrett\Documents\Projects\MNIST_Kaggle\MNIST_digits\data\\test.csv")
print(test.shape)
test.head()

