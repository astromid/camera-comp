import utils
from keras.models import Model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.applications.resnet50 import ResNet50


def resnet50():
    # i = Input(shape=(None, None, 3))
    # norm_i = BatchNormalization()(i)
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    # x = base_model(norm_i)
    x = base_model.output
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    out = Dense(units=utils.N_CLASS, activation='softmax')(x)
    # model = Model(inputs=i, outputs=out)
    model = Model(inputs=base_model.input, outputs=out)
    for layer in base_model.layers:
        layer.trainable = False
    return model
