import utils
from keras.models import Model
from keras.layers import Dense, Dropout, Input, concatenate
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121


def resnet50():
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    manip_flags = Input(shape=(None, 1))
    x = base_model.output
    x = concatenate([x, manip_flags])
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    out = Dense(units=utils.N_CLASS, activation='softmax')(x)
    model = Model(inputs=(base_model.input, manip_flags), outputs=out)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def densenet121():
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    x = base_model.output
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    out = Dense(units=utils.N_CLASS, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=out)
    for layer in base_model.layers:
        layer.trainable = False
    return model
