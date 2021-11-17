import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("imported successfully")



if __name__ == "__main__":
    # train and validation directory
    train_dir = "dataset/training/"
    validation_dir = "dataset/validation"
    train_lion_dir = "dataset/training/lion"               # Directory with our training lion pictures
    train_non_lion_dir = "dataset/training/non_lion"       # Directory with our training non_lion pictures
    validation_lion_dir = "dataset/training/lion"          # Directory with our validation lion pictures
    validation_non_lion_dir = "dataset/training/non_lion"  # Directory with our validation non_lion pictures

    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1. / 255., rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    train_generator = train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode='binary',
                                                        target_size=(224, 224))

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(validation_dir, batch_size=20, class_mode='binary',
                                                            target_size=(224, 224))
    from tensorflow.keras.applications.vgg16 import VGG16

    base_model = VGG16(input_shape=(224, 224, 3),  # Shape of our images
                       include_top=False,  # Leave out the last fully connected layer
                       weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(base_model.output)

    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)

    # Add a final sigmoid layer with 1 node for classification output
    x = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(base_model.input, x)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
    print("training started")
    vgghist = model.fit(train_generator, validation_data=validation_generator,
                        steps_per_epoch=70,
                        epochs=12)



    model.save('vgg16_new.h5')

    print("model saved successfully")

    # fig=pd.DataFrame(vgghist.history)
    # fig.save('train_vs_validation.pdf')
