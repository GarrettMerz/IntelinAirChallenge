import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten 
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, ELU, Conv3D, ConvLSTM2D 
from tensorflow.keras.layers import UpSampling2D, Concatenate, Multiply
from tensorflow.keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import random
import cv2
import sys
from config import *
from models import *
from losses import *
from models import *
from get_data import *

def tversky_loss(targets, inputs, alpha=1, beta=1, numLabels=2):

    tversky = 0
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
        
    #True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
    TN = K.sum(((1-inputs) * (1-targets)))    
    
    #Sum over both classes
    tversky= ((TP) / (TP + alpha*FP + beta*FN))+ ((TN) / (TN + alpha*FN + beta*FP)) 
    return (2 - tversky)

    
def focal_loss(targets, inputs, numLabels=2):

    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    inputs2 = K.flatten(1-inputs)
    targets2 = K.flatten(1-targets)

    #BCE for class 0
    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)

    #BCE for class 1
    BCE2 = K.binary_crossentropy(targets2, inputs2)
    BCE_EXP2 = K.exp(-BCE2)           

    #Sum over both classes
    focal_loss = K.mean(K.pow((1-BCE_EXP), 2) * BCE)+K.mean(K.pow((1-BCE_EXP2), 2) * BCE2)

    return focal_loss
            
def hybrid_loss(targets, inputs, _lambda_ = 1):
    tversky= tversky_loss(targets, inputs)
    focal=focal_loss(targets, inputs)    
    result = tversky + _lambda_ * focal
    return result

def total_loss(y_true, y_pred):
    sumval = hybrid_loss(y_true, y_pred)
    return (1/3)*sumval
