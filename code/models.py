import utils
from keras.models import Model
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121


def resnet50():
    i = Input(shape=(None, None, 3))
    bn_i = BatchNormalization()(i)
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    x = base_model(bn_i)
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    out = Dense(units=utils.N_CLASS, activation='softmax')(x)
    model = Model(inputs=i, outputs=out)
    for layer in base_model.layers:
        layer.trainable = False
    return model


def densenet121():
    i = Input(shape=(None, None, 3))
    bn_i = BatchNormalization()(i)
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    x = base_model(bn_i)
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    out = Dense(units=utils.N_CLASS, activation='softmax')(x)
    model = Model(inputs=i, outputs=out)
    for layer in base_model.layers:
        layer.trainable = False
    return model
