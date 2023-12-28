import os
import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, image_utils
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Dropout, Conv2D, MaxPooling2D, AveragePooling2D
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

DIR = "gender-age/UTKFace"

class train_gender:
    def __init__(self):
        self.image_paths = []
        self.age_labels = []
        self.gender_labels = []
        self.race_labels = []

    def csv_file(self, DIR):
        # tqdm is a library in Python which is used for creating Progress Meters or Progress Bars
        for filename in tqdm(os.listdir(DIR)):
            image_path = os.path.join(DIR, filename)
            temp = filename.split('_')
            age = int(temp[0])
            # is an integer from 0 to 116, indicating the age
            gender = int(temp[1])
            #  is either 0 (male) or 1 (female)
            # race = int(temp[2])
            #  from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).\
            self.image_paths.append(image_path)
            self.age_labels.append(age)
            self.gender_labels.append(gender)

        return self.image_paths, self.age_labels, self.gender_labels

    # extraction of all images into numpy array
    def extract_features(self, images_path):
        features = []
        for image in tqdm(images_path):
            img = image_utils.load_img(path=image, grayscale=True)
            img = img.resize((48, 48))
            feature = np.array(img)
            # features.append(img)

            # feature = np.array(feature)
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], 1))
            features.append(feature)
        return features


    def identity_block(self, X, level: int, filters: list[int], block: int):
        conv_name = f'conv{level}_{block}' + '_{layer}_{type}'

        f1, f2, f3 = filters

        X_shortcut = X

        X = Conv2D(filters=f1, kernel_size=(1, 1), padding='valid', name=conv_name.format(layer=1, type='conv'))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=2, type='bn'))(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=f2, kernel_size=(1, 1), padding='same', name=conv_name.format(layer=3, type='conv'))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=4, type='bn'))(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=f3, kernel_size=(1, 1), padding='valid', name=conv_name.format(layer=5, type='conv'))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=6, type='bn'))(X)

        X = Add()([X, X_shortcut])

        X = Activation('relu')(X)

        return X

    def convolutional_block(self, X, level: int, filters: list[int], block: int, s: tuple[int, int] = (2, 2)):

        conv_name = f'conv{level}_{block}' + '_{layer}_{type}'

        f1, f2, f3 = filters

        X_shortcut = X

        X = Conv2D(filters=f1, kernel_size=(1, 1), strides=s, padding='valid',
                   name=conv_name.format(layer=1, type='conv'))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=2, type='bn'))(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=f2, kernel_size=(1, 1), strides=(1, 1), padding='same',
                   name=conv_name.format(layer=3, type='conv'))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=4, type='bn'))(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   name=conv_name.format(layer=5, type='conv'))(X)
        X = BatchNormalization(axis=3, name=conv_name.format(layer=6, type='bn'))(X)

        X_shortcut = Conv2D(f3, kernel_size=(1, 1), strides=s, name=conv_name.format(layer=7, type='conv'))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=conv_name.format(layer=8, type='bn'))(X_shortcut)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def resnet50(self, input_shape: tuple[int, int, int], classes=3):
        X_input = Input(input_shape)

        X = ZeroPadding2D((3, 3))(X_input)

        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)  # 64 filters of 7*7
        X = BatchNormalization(axis=3, name='bn_conv1')(X)  # batchnorm applied on channels
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        X = self.convolutional_block(X=X, level=2, filters=[64, 64, 256], block=1, s=(1, 1))
        X = self.identity_block(X, level=2, filters=[64, 64, 256], block=2)
        X = self.identity_block(X, level=2, filters=[64, 64, 256], block=3)

        X = self.convolutional_block(X, level=3, filters=[64, 64, 256], block=1, s=(2, 2))
        X = self.identity_block(X, level=3, filters=[64, 64, 256], block=2)
        X = self.identity_block(X, level=3, filters=[64, 64, 256], block=3)
        X = self.identity_block(X, level=3, filters=[64, 64, 256], block=4)

        X = self.convolutional_block(X, level=4, filters=[64, 64, 256], block=1, s=(2, 2))
        X = self.identity_block(X, level=4, filters=[64, 64, 256], block=2)
        X = self.identity_block(X, level=4, filters=[64, 64, 256], block=3)
        X = self.identity_block(X, level=4, filters=[64, 64, 256], block=4)
        X = self.identity_block(X, level=4, filters=[64, 64, 256], block=5)
        X = self.identity_block(X, level=4, filters=[64, 64, 256], block=6)

        X = self.convolutional_block(X, level=5, filters=[64, 64, 256], block=1, s=(2, 2))
        X = self.identity_block(X, level=5, filters=[64, 64, 256], block=2)
        X = self.identity_block(X, level=5, filters=[64, 64, 256], block=3)

        X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

        X = Flatten()(X)

        dense_1 = Dense(256, activation='softmax')(X)
        dense_2 = Dense(256, activation='softmax')(X)

        dropout_1 = Dropout(0.3)(dense_1)
        dropout_2 = Dropout(0.3)(dense_2)

        output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
        output_2 = Dense(1, activation='softmax', name='age_out')(dropout_2)

        model = Model(inputs=X_input, outputs=[output_1, output_2], name='ResNet50')

        model.compile(optimizer='adam',
                      loss={'gender_out': 'binary_crossentropy', 'age_out': 'mean_squared_error'},
                      metrics={'gender_out': 'accuracy', 'age_out': 'mae'})
        # Fit the model
        model.fit(
            X_train,
            {'gender_out': y_gender_train, 'age_out': y_age_train},
            validation_data=(X_val, {'gender_out': y_gender_val, 'age_out': y_age_val}),
            batch_size=32,
            epochs=10,
            verbose=1
        )

        return model

gender_ = train_gender()
df = pd.DataFrame()
df['image_paths'], df['age'], df['gender'] = gender_.csv_file(DIR=DIR)
features = gender_.extract_features(df['image_paths'])
print(features[0].shape)
# print(features[0])

features = np.array(features)
age = np.array(df['age'])
gender = np.array(df['gender'])
X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val = train_test_split(
    features, age, gender, test_size=0.2, random_state=0)
#
history = gender_.resnet50((48, 48, 1))
print(history)
# input_shape = (48, 48, 1)
# num_classes = 2
    # gender_model = ResNet50(input_shape, classes=num_classes)


# model_json = gender_model.to_json()
# with open("gender_model2.json", "w") as json_file:
#     json_file.write(model_json)
#
#     # save trained model weight in .h5 file
# gender_model.save_weights('gender_model2.h5')

