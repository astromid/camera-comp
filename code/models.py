import utils
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.resnet50 import ResNet50


def resnet50():
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    x = base_model.output
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    out = Dense(units=utils.N_CLASS, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=out)
    return model
