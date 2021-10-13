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
from src.config import *

def UNet(image_input, boundary_input):
    #The U-Net block. Given that I'm training three U-Nets in parallel, I can only afford a few filters for each layer.

    #Mask inputs with boundary mask
    Mask_in0 = Multiply()([boundary_input,image_input[:,:,:,0]])
    Mask_in1 = Multiply()([boundary_input,image_input[:,:,:,1]])
    Mask_in2 = Multiply()([boundary_input,image_input[:,:,:,2]])
    masked_input = Concatenate()([Mask_in0,Mask_in1,Mask_in2])
   
    #Encoder layers
    c1 = Conv2D(8, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(masked_input)
    c1 = ELU()(c1)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(8, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(c1)
    c1 = ELU()(c1)
    c1 = BatchNormalization()(c1)
    
    p1 = MaxPooling2D((2,2))(c1)

    c2 = Conv2D(8*2, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(p1)
    c2 = ELU()(c2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(8*2, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(c2)
    c2 = ELU()(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2,2))(c2)
    

    c3 = Conv2D(8*4, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(p2)
    c3 = ELU()(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(8*2, kernel_size=(1,1), kernel_initializer='he_normal', padding = 'same')(c3)
    c3 = ELU()(c3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(8*4, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(c3)
    c3 = ELU()(c3)
    c3 = BatchNormalization()(c3)

    p3 = MaxPooling2D((2,2))(c3)

    c4 = Conv2D(8*6, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(p3)
    c4 = ELU()(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(8*4, kernel_size=(1,1), kernel_initializer='he_normal', padding = 'same')(c4)
    c4 = ELU()(c4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(8*6, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(c4)
    c4 = ELU()(c4)
    c4 = BatchNormalization()(c4)

    p4 = MaxPooling2D((2,2))(c4)

    c5 = Conv2D(8*6, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(p4)
    c5 = ELU()(c5)
    c5 = BatchNormalization()(c5)

    #Decoder Layers

    u1 = UpSampling2D((2,2))(c5)
    concat1 = Concatenate()([c4, u1])

    c6 = Conv2D(8*4, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(concat1)
    c6 = ELU()(c6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(8*4, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(c6)
    c6 = ELU()(c6)
    c6 = BatchNormalization()(c6)


    u2 = UpSampling2D((2,2))(c6)
    concat2 = Concatenate()([c3, u2])

    c7 = Conv2D(8*2, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(concat2)
    c7 = ELU()(c7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(8*2, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(c7)
    c7 = ELU()(c7)
    c7 = BatchNormalization()(c7)

    u3 = UpSampling2D((2,2))(c7)
    concat3 = Concatenate()([c2, u3])

    c8 = Conv2D(8, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(concat3)
    c8 = ELU()(c8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(8, kernel_size=(3,3), kernel_initializer='he_normal', padding = 'same')(c8)
    c8 = ELU()(c8)
    c8 = BatchNormalization()(c8)

    u4 = UpSampling2D((2,2))(c8)
    concat4 = Concatenate()([c1, u4])

    c9 = Conv2D(4, kernel_size = (1,1), kernel_initializer='he_normal', padding = 'same')(concat4)
    c9 = ELU()(c9)
    c9 = BatchNormalization()(c9)

    mask_out = Conv2D(1, (1,1), kernel_initializer='he_normal', padding = 'same', activation = 'sigmoid')(c9)

    #Apply boundary mask again to output
    multmask = Multiply()([mask_out, boundary_input])

    return multmask

def ConvLSTM(movieIn, boundary_input):

    #Convolutional LSTM module.
    x = ConvLSTM2D(
    filters=16,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="elu",
    )(movieIn)

    x = BatchNormalization()(x)

    x = ConvLSTM2D(
    filters=16,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="elu",
    )(x)
    
    x = BatchNormalization()(x)
    
    x = ConvLSTM2D(
    filters=16,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="elu",
    )(x)
    
    x = BatchNormalization()(x)
    
    lastmask = Conv3D(
    filters=1,
    kernel_size=(3, 3, 3),
    activation="sigmoid",
    padding="same"
    )(x)
    
    #Mask the output at each timestep, return as a three-channel image
    P0 = Multiply()([boundary_input,lastmask[:,0,:,:,:]])
    P1 = Multiply()([boundary_input,lastmask[:,1,:,:,:]])
    P2 = Multiply()([boundary_input,lastmask[:,2,:,:,:]])
    
    outmask = Concatenate()([P0,P1,P2])
    
    return outmask

def LastOnlyModel():
    #Dummy model to only use the last timestep (not employed)

    image0 = Input(shape=(HEIGHT, WIDTH, CHANNELS), name = 'img0_input')
    image1 = Input(shape=(HEIGHT, WIDTH, CHANNELS), name = 'img1_input')
    image2 = Input(shape=(HEIGHT, WIDTH, CHANNELS), name = 'img2_input')
    boundary = Input(shape=(HEIGHT, WIDTH, 1), name = 'boundary_input')
    outmask= UNet(image2, boundary)
    
    model = Model(inputs = [image0,image1,image2, boundary], outputs = outmask, name='SemanticSegModelOnlyLast')

    return model

def FullModel():
    #The full model. Three U-Nets, one Concatenate, and a ConvLSTM.

    image0 = Input(shape=(HEIGHT, WIDTH, CHANNELS), name = 'img0_input')
    image1 = Input(shape=(HEIGHT, WIDTH, CHANNELS), name = 'img1_input')
    image2 = Input(shape=(HEIGHT, WIDTH, CHANNELS), name = 'img2_input')
    boundary = Input(shape=(HEIGHT, WIDTH, 1), name = 'boundary_input')
    
    X1= UNet(image0, boundary)
    X2= UNet(image1, boundary)
    X3= UNet(image2, boundary)

    #Arrange U-Net outputs along a time dimension
    X1 = tf.keras.backend.expand_dims(X1,axis=1)
    X2 = tf.keras.backend.expand_dims(X2,axis=1)
    X3 = tf.keras.backend.expand_dims(X3,axis=1)
    movie=Concatenate(axis=1)([X1, X2, X3])

    outmask = ConvLSTM(movie, boundary)
    print(outmask.shape)                   
    model = Model(inputs = [image0,image1,image2, boundary], outputs = outmask, name='SemanticSegModel')

    return model

def lrfn(epoch):
    #learning rate decay function
    if epoch > 15:
        return 2e-6
    elif epoch > 25:
        return 1e-6
    return 1e-6

