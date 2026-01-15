import os
import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras


from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from pathlib import Path
import numpy as np

BATCH_SIZE = 64

train_generator = ImageDataGenerator(rotation_range=90, 
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input) # VGG16 preprocessing

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input) # VGG16 preprocessing

# Working on Data cleaning
train_data_dir = 'XXXXXXXXXXXXXX'
test_data_dir = 'XXXXXXXXXXXXXXXXX'

traingen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               classes=13,
                                               subset='training',
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               seed=42)

testgen = test_generator.flow_from_directory(test_data_dir,
                                             target_size=(224, 224),
                                             class_mode=None,
                                             classes=13,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=42)

'''
Model-1: Non-trinable Layers
'''

def model_a(input_shape, n_classes, optimizer):
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    base_model = VGG16(weights='imagenet', include_top=False,input_shape=input_shape)
    
    # Freezing Base Model's Layer as Non-trainable
    for layer in base_model.layers:
        layer.trainable = False
    
    # Creating a new model on top of the base mode (i:e Fully Connected Layers)
    top_model = base_model.output
    top_model = Flatten(name="flatten")(top_model)
    # a.512 Dense Layer along with ReLu Activation Function
    top_model = Dense(512, activation='relu')(top_model)
    # b.Drop Out Layer
    top_model = Dropout(0.6)(top_model)
    # c. 128 Dense Layer along with ReLu Activation Function
    top_model = Dense(128, activation='relu')(top_model)
    # d.Batch Normalization
    top_model = BatchNormalization()(top_model)
    # e.Drop Out Layer
    top_model = Dropout(0.4)(top_model)
    # f.64 Dense Layer along with ReLu Activation Function
    output_layer = Dense(64, activation='relu')(top_model)

    #output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model, output_layer

'''
Model-2: With Trainable Layers
'''

def model_b(input_shape, n_classes, optimizer):
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    base_model = VGG16(weights='imagenet', include_top=False,input_shape=input_shape)
    
    # Freezing Base Model's Layer as Non-trainable
    for layer in base_model.layers:
        layer.trainable = True
    
    # Creating a new model on top of the base mode (i:e Fully Connected Layers)
    top_model = base_model.output
    top_model = Flatten(name="flatten")(top_model)
    # a.512 Dense Layer along with ReLu Activation Function
    top_model = Dense(512, activation='relu')(top_model)
    # b.Drop Out Layer
    top_model = Dropout(0.6)(top_model)
    # c. 128 Dense Layer along with ReLu Activation Function
    top_model = Dense(128, activation='relu')(top_model)
    # d.Batch Normalization
    top_model = BatchNormalization()(top_model)
    # e.Drop Out Layer
    top_model = Dropout(0.4)(top_model)
    # f.64 Dense Layer along with ReLu Activation Function
    output_layer = Dense(64, activation='relu')(top_model)

    #output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=base_model.input, outputs=output_layer)
    
    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model



