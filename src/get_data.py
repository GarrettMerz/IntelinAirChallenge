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

def make_df(in_dir = src_dir):
    #Function to make a pandas dataframe from a directory containing subdirectories of images
    input_img0_names = []
    input_img1_names = []
    input_img2_names = []
    boundary_mask_names = []
    nutrient_mask_names = []
    for dir in os.listdir(in_dir):
        input_img0_names.append(os.path.join(src_dir, f'{dir}/image_i0.png'))
        input_img1_names.append(os.path.join(src_dir, f'{dir}/image_i1.png'))
        input_img2_names.append(os.path.join(src_dir, f'{dir}/image_i2.png'))
        boundary_mask_names.append(os.path.join(src_dir, f'{dir}/bounday_mask.png'))
        nutrient_mask_names.append(os.path.join(src_dir, f'{dir}/nutrient_mask_g0.png'))

    df = pd.DataFrame({'image_0':input_img0_names , 'image_1':input_img1_names, 'image_2':input_img2_names, 
                   'boundary_mask':boundary_mask_names, 'nutrient_mask':nutrient_mask_names})

    return df

def make_traintest(df):
    #Function to split a dataframe containing information about images into train, val and test sets
    XY_train,XY_valtest = train_test_split(df, train_size=0.8, random_state=0)
    XY_val, XY_test = train_test_split(XY_valtest, train_size=0.5, random_state=0)
    XY_train.reset_index(drop=True, inplace=True)
    XY_val.reset_index(drop=True, inplace=True)
    XY_test.reset_index(drop=True, inplace=True)

    return XY_train,XY_val,XY_test

def myGenerator(generator1,generator2,generator3,boundgenerator,outgenerator):
    #Custom generator function. This returns the three input images, the boundary mask (reduced to one channel) and the nutrient mask.
        while True:
            for x0,x1,x2,bound,y1 in zip(generator1,generator2,generator3,boundgenerator,outgenerator):
                yield ([x0,x1,x2,bound[:,:,:,2]], y1)


def df_to_generator(XY_df, do_augments = False):
    #Flow from a specified dataframe with a myGenerator.
    image0_datagen = ImageDataGenerator(rescale=1./255.)
    image1_datagen = ImageDataGenerator(rescale=1./255.)
    image2_datagen = ImageDataGenerator(rescale=1./255.)
    mask_datagen = ImageDataGenerator(rescale=1./255.)
    boundary_datagen = ImageDataGenerator(rescale=1./255.)

    seed = 1

    image0_generator = image0_datagen.flow_from_dataframe(
        dataframe=XY_df,
        x_col="image_0",
        class_mode=None,
        horizontal_flip=do_augments,
        vertical_flip=do_augments,
        target_size = (WIDTH, HEIGHT),
        batch_size = BATCH_SIZE,
        seed=seed)

    image1_generator = image1_datagen.flow_from_dataframe(
        dataframe=XY_df,
        x_col="image_1",
        class_mode=None,
        horizontal_flip=do_augments,
        vertical_flip=do_augments,
        target_size = (WIDTH, HEIGHT),
        batch_size = BATCH_SIZE,
        seed=seed)

    image2_generator = image2_datagen.flow_from_dataframe(
        dataframe=XY_df,
        x_col="image_2",
        class_mode=None,
        horizontal_flip=do_augments,
        vertical_flip=do_augments,
        target_size = (WIDTH, HEIGHT),
        batch_size = BATCH_SIZE,
        seed=seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        dataframe=XY_df,
        x_col="nutrient_mask",
        class_mode=None,
        horizontal_flip=do_augments,
        vertical_flip=do_augments,
        target_size = (WIDTH, HEIGHT),
        batch_size = BATCH_SIZE,
        seed=seed)

    boundary_generator = boundary_datagen.flow_from_dataframe(
        dataframe=XY_df,
        x_col="boundary_mask",
        class_mode=None,
        horizontal_flip=do_augments,
        vertical_flip=do_augments,
        target_size = (WIDTH, HEIGHT),
        batch_size = BATCH_SIZE,
        seed=seed)

    # combine generators into one which yields image and masks
    df_generator = myGenerator(image0_generator, image1_generator,
                                  image2_generator, boundary_generator, mask_generator)

    return df_generator
