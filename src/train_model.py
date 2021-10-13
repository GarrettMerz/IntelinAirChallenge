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
from src.models import *
from src.losses import *
from src.get_data import *

def train_model():
    #Get the training and validation set
    df = make_df()
    XY_train,XY_val,XY_test = make_traintest(df)
    train_generator = df_to_generator(XY_train,do_augments=True)
    val_generator = df_to_generator(XY_val,do_augments=False)
    train_steps = len(XY_train)
    val_steps = len(XY_val)

    # Learning rate decay function
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: lrfn(step))
    opt_adam = optimizers.Adam()
    # Model checkpoint, saves weights if val loss reduces
    checkpoint = tf.keras.callbacks.ModelCheckpoint('models/model.h5', 'val_loss', save_best_only=True, verbose=1)

    #load and compile model
    model = FullModel()
    model.compile(optimizer = opt_adam, loss = hybrid_loss, metrics = [tf.keras.metrics.MeanIoU(num_classes=2)])
    model.summary()

    #fit model to train set, validate on validation set. Save best model after 5 epochs
    history = model.fit(train_generator, validation_data=val_generator, 
                        steps_per_epoch=train_steps, validation_steps=val_steps, 
                        epochs = 20, verbose=1, callbacks=[checkpoint, lr_callback])


    #Plot loss and IOU curves
    plt.figure()
    plt.plot(history.history['mean_io_u'], label = 'Train IOU')
    plt.plot(history.history['val_mean_io_u'], label = 'Val IOU')
    plt.title('Mean IOU')
    plt.legend()
    plt.savefig("output_figs/metrics/IOU_curve.png", dpi=150)


    plt.figure()
    plt.plot(history.history['loss'], label = 'Train loss')
    plt.plot(history.history['val_loss'], label = 'Val loss')
    plt.title('loss')
    plt.legend()
    plt.savefig("output_figs/metrics/loss_curve.png", dpi=150)
    return

if __name__ == "__main__":
    train_model()
