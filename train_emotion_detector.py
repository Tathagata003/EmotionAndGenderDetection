from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
import cv2
import tensorflow as tf

EXPS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class train_gender:
    def __init__(self, train_path: str, validation_path: str):
        # Initialize image data generator with rescaling
        self.train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                            rotation_range=20,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest')
        self.validation_data_gen = ImageDataGenerator(rescale=1. / 255)

        # Preprocess all train images
        self.train_generator = self.train_data_gen.flow_from_directory(
            train_path,
            target_size=(48, 48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

        # Preprocess all test images
        self.validation_generator = self.validation_data_gen.flow_from_directory(
            validation_path,
            target_size=(48, 48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    def model(self):
        # create model structure
        emotion_model = Sequential()

        emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(0.25))

        emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        emotion_model.add(Dropout(0.25))

        emotion_model.add(Flatten())
        emotion_model.add(Dense(1024, activation='relu'))
        emotion_model.add(Dropout(0.25))
        emotion_model.add(Dense(7, activation='softmax'))

        # cv2.ocl.setUseOpenCL(False)

        emotion_model.compile(loss='categorical_crossentropy',
                              optimizer=Adam,
                              metrics=['accuracy']
                              )

        # Train the neural network/model
        emotion_model_info = emotion_model.fit_generator(
            self.train_generator,
            steps_per_epoch=28709 // 64,
            epochs=10,
            validation_data=self.validation_generator,
            validation_steps=7178 // 64)

        # save model structure in jason file
        model_json = emotion_model.to_json()
        with open("emotion_model.json", "w") as json_file:
            json_file.write(model_json)

        # save trained model weight in .h5 file
        emotion_model.save_weights('emotion_model.h5')

        return emotion_model_info

