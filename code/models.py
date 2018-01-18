from keras.models import Model
from keras.layers import Conv2D, Dense
from keras.layers import Activation, Input, Flatten
from keras.layers import Multiply, Add
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization
from keras.applications.resnet50 import ResNet50

N_CLASS = 10


def resnet50():
    i = Input(shape=(512, 512, 3))
    res = ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )(i)
    dense1 = Dense(units=2048, activation='relu')(res)
    drp1 = Dropout(rate=0.2)(dense1)
    out = Dense(units=N_CLASS, activation='softmax')(drp1)
    model = Model(inputs=i, outputs=out)
    return model
