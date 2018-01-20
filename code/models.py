import utils
from keras.models import Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.applications.resnet50 import ResNet50


def resnet50():
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    norm_inp = BatchNormalization()(base_model.input)
    x = base_model.output
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    out = Dense(units=utils.N_CLASS, activation='softmax')(x)
    model = Model(inputs=norm_inp, outputs=out)
    for layer in base_model.layers:
        layer.trainable = False
    return model
