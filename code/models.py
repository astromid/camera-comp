import utils
from keras.applications import *
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape, concatenate
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Multiply, Add
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras_contrib.layers import BatchRenormalization


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


def pretrained_model(clf_name):
    image, manip_flag = _inputs()
    module_name = utils.CLF2MODULE[clf_name]
    class_name = utils.CLF2CLASS[clf_name]
    base_model_class = getattr(globals()[module_name], class_name)
    print(f'Using {class_name} as base model')
    base_model = base_model_class(
        include_top=False,
        weights='imagenet',
        pooling='avg')
    x = base_model(image)
    out = _top(x, manip_flag)
    model = Model(inputs=(image, manip_flag), outputs=out)
    for layer in model.get_layer(clf_name).layers:
        layer.trainable = False
    return model


def train_pretrained_model(clf_name, model, train, val, model_args, f_epochs, epochs, cb_f, cb_e):
    if f_epochs != 0:
        # train with frozen pretrained block
        model.compile(**model_args)
        model.summary()
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
        for layer in model.get_layer(clf_name).layers:
            layer.trainable = True
        model.compile(**model_args)
        model.summary()
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


class SeResNet3:

    def __init__(self):
        image, manip_flag = _inputs()
        x = BatchRenormalization()(image)

        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x)
        x = BatchRenormalization()(x)
        x = Activation(activation='relu')(x)
        x = self.resblock(z=x, n_in=16, n_out=16)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Dropout(rate=0.1)(x)

        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = BatchRenormalization()(x)
        x = Activation(activation='relu')(x)
        x = self.resblock(z=x, n_in=32, n_out=32)
        x = self.resblock(z=x, n_in=32, n_out=32)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        x = BatchRenormalization()(x)
        x = Activation(activation='relu')(x)
        x = self.resblock(z=x, n_in=64, n_out=64)
        x = self.resblock(z=x, n_in=64, n_out=64)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
        x = BatchRenormalization()(x)
        x = Activation(activation='relu')(x)
        x = self.resblock(z=x, n_in=128, n_out=128)
        x = self.resblock(z=x, n_in=128, n_out=128)
        x = Dropout(rate=0.2)(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        x = BatchRenormalization()(x)
        x = Activation(activation='relu')(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(units=256, activation='relu')(x)
        x = Dropout(rate=0.2)(x)
        out = Dense(units=utils.N_CLASS, activation='softmax')(x)

        model = Model(inputs=(image, manip_flag), outputs=out)
        self.model = model

    @staticmethod
    def scale(z, n, red=16):
        x = Conv2D(filters=red, kernel_size=(1, 1), activation='relu')(z)
        x = Conv2D(filters=n, kernel_size=(1, 1))(x)
        return Activation(activation='sigmoid')(x)

    def resblock(self, z, n_in, n_out):
        x = Conv2D(n_in, (3, 3), padding='same')(z)
        x = BatchRenormalization()(x)
        x = Activation(activation='relu')(x)
        x = Conv2D(n_out, (3, 3), padding='same')(x)
        x = BatchRenormalization()(x)
        scale = self.scale(x, n_out)
        x = Multiply()([scale, x])
        x = Add()([z, x])
        return Activation(activation='relu')(x)
