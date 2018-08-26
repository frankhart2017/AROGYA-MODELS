from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

img_width, img_height = 128, 128
train_data_dir = "img/train"
validation_data_dir = "img/validation"
batch_size = 32
epochs = 5

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(2048, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(7, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
rescale = 1./255)

test_datagen = ImageDataGenerator(
rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("models/vgg19_1_chole.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Train the model 
model_final.fit_generator(
train_generator,
epochs = epochs,
validation_data = validation_generator,
verbose=1,
use_multiprocessing=True,
shuffle=True,
callbacks = [checkpoint, early])

from keras.models import load_model
model_final = load_model('vgg16_1.h5')

model_final.fit_generator(
train_generator,
epochs = epochs,
validation_data = validation_generator,
verbose=1,
callbacks= [checkpoint, early])

import cv2
import numpy as np
im = cv2.imread('apple.jpg')
im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (128, 128)).astype(np.float32) / 255.0
im = np.expand_dims(im, axis =0)
print(np.round(model_final.predict(im)))

train_generator.class_indices