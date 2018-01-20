import utils
from keras.models import Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.applications.resnet50 import ResNet50


def resnet50():
    i = BatchNormalization(input_shape=(512, 512, 3))
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=i,
        pooling='avg'
    )
    x = base_model.output
    x = Dense(units=2048, activation='relu')(x)
    x = Dropout(rate=0.2)(x)
    out = Dense(units=utils.N_CLASS, activation='softmax')(x)
    model = Model(inputs=i, outputs=out)
    for layer in base_model.layers:
        layer.trainable = False
    return model
