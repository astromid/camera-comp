import utils
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet201
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape, concatenate
from keras.layers import Conv2D
from keras.layers import Activation, Flatten
from keras.layers import Multiply, Add
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout
# from keras_contrib.layers import BatchRenormalization


def _inputs():
    image = Input(shape=(utils.CROP_SIDE, utils.CROP_SIDE, 3))
    manip_flag = Input(shape=(1,))
    return image, manip_flag


def _top(x, manip_flag):
    x = Reshape((-1,))(x)
    x = concatenate([x, manip_flag])
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    out = Dense(units=utils.N_CLASS, activation='softmax')(x)
    return out


def resnet50():
    image, manip_flag = _inputs()
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg')
    x = base_model(image)
    out = _top(x, manip_flag)
    model = Model(inputs=(image, manip_flag), outputs=out)
    for layer in model.get_layer('resnet50').layers:
        layer.trainable = False
    return model


def densenet201():
    image, manip_flag = _inputs()
    base_model = DenseNet201(
        include_top=False,
        weights='imagenet',
        pooling='avg')
    x = base_model(image)
    out = _top(x, manip_flag)
    model = Model(inputs=(image, manip_flag), outputs=out)
    for layer in model.get_layer('densenet201').layers:
        layer.trainable = False
    return model


def xception():
    image, manip_flag = _inputs()
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        pooling='avg')
    x = base_model(image)
    out = _top(x, manip_flag)
    model = Model(inputs=(image, manip_flag), outputs=out)
    for layer in model.get_layer('xception').layers:
        layer.trainable = False
    return model


def train_model(model, train, val, model_args, f_epochs, epochs, cb_f, cb_e):
    if f_epochs != 0:
        # train with frozen pretrained block
        model.compile(**model_args)
        model.fit_generator(
            generator=train,
            steps_per_epoch=len(train),
            epochs=f_epochs,
            verbose=0,
            callbacks=cb_f,
            validation_data=val,
            validation_steps=len(val))
    if epochs > f_epochs:
        # defrost pretrained block
        for layer in model.get_layer('densenet201').layers:
            layer.trainable = True
        model.compile(**model_args)
        model.fit_generator(
            generator=train,
            steps_per_epoch=len(train),
            epochs=epochs,
            verbose=0,
            callbacks=cb_e,
            validation_data=val,
            validation_steps=len(val),
            initial_epoch=f_epochs)
    return model
