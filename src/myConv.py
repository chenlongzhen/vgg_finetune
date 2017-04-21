# -*- coding: utf-8 -*-
"""myConv model for Keras.

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape


def myConv(include_top=True,input_tensor=None, input_shape=None,classes=1):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top # When setting`include_top=True `input_shape` should be ' + str(default_shape)
                                      )
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
#    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

#    # Block 4
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
#
#    # Block 5
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(classes,activation='sigmoid')(x)

#    x = Flatten(name='flatten')(x)
#    x = Dense(4096, activation='relu', name='fc1')(x)
#    x = Dense(4096, activation='relu', name='fc2')(x)
#    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='myConv')

    return model
