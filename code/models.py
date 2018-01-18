from keras.models import Model
from keras.layers import Conv2D, Dense
from keras.layers import Activation, Input, Flatten
from keras.layers import Multiply, Add
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization
from keras import optimizers, losses
from keras.activations import relu, softmax
from keras.metrics import categorical_accuracy
from keras.applications.resnet50 import ResNet50

N_CLASS = 10

