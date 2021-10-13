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
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import random
import cv2
import sys
from config import *
from models import *
from losses import *
from get_data import *

df= make_df()
XY_train,XY_val,XY_test=make_traintest(df)
print(XY_test.iloc[0,0])

test_generator = df_to_generator(XY_test, do_augments=False)
test_size = get_gen_size(XY_test, do_augments=False)
test_iter = iter(test_generator)

# Load the model
model = load_model(
    "models/model.h5",
    custom_objects={'hybrid_loss':hybrid_loss},
    compile=True
)

loss = model.evaluate(test_generator, steps=test_size, verbose = 1)

for i in range(test_size):
    images, masks = next(test_iter)

    pred_masks = model.predict(images)[:,:,:,2]
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    for a in ax:
        a.axis('off')

    ax[0].imshow(images[2][0])
    ax[0].set_title('Original')
    ax[1].imshow(pred_masks[0])
    ax[1].set_title('Prediction')
    ax[2].imshow(masks[0,:,:,2])
    ax[2].set_title('Ground truth')

    plt.savefig(f"output_figs/test_eval/pred_{i}.png", dpi=150)
    plt.close()