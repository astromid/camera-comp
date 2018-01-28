import utils
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape, concatenate
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet201


def resnet50():
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        pooling='avg')
    manip_flags = Input(shape=(1,))
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


def densenet201():
    base_model = DenseNet201(
        include_top=False,
        weights='imagenet',
        pooling='avg')
    manip_flags = Input(shape=(1,))
    x = base_model.output
    x = Reshape((-1,))(x)
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
        for layer in model.layers:
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
