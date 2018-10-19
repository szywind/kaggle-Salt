from keras.layers import Conv2D
from keras.layers import Activation
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten
from constants import INPUT_WIDTH, INPUT_HEIGHT

from .blocks import Transpose2D_block
from .blocks import Upsample2D_block
from ..utils import get_layer_number, to_tuple

from keras.layers import UpSampling2D, Dropout, BatchNormalization, AveragePooling2D
from keras.layers import concatenate

import keras.backend as K

def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'glorot_uniform',
        'use_bias': False,
        'padding': 'same',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        'axis': 3,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params

def build_unet(backbone, classes, skip_connection_layers,
               decoder_filters=(256,128,64,32,16),
               upsample_rates=(2,2,2,2,2),
               n_upsample_blocks=5,
               block_type='upsampling',
               activation='sigmoid',
               use_batchnorm=True):

    input = backbone.input
    x = backbone.output


    # TODO

    # image_pool = GlobalAveragePooling2D()(x)
    image_pool = AveragePooling2D(pool_size=4)(x)

    image_pool = Conv2D(64, (1,1))(image_pool)

    # classify = Flatten()(image_pool)
    classify = Dense(1, activation='sigmoid')(image_pool)
    up_image_pool = UpSampling2D(size=(INPUT_HEIGHT, INPUT_WIDTH))(image_pool)

    # conv_params = get_conv_params()
    # bn_params = get_bn_params()
    # channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # filters = x._keras_shape[channel_axis]
    #
    # x = Conv2D(filters, (3, 3), strides=(1, 1), name='conv0', **conv_params)(x)
    # x = BatchNormalization(name='bn0', **bn_params)(x)
    # x = Activation('relu', name='relu0')(x)
    # x = Conv2D(filters, (3, 3), strides=(1, 1), name='conv0', **conv_params)(x)
    # x = BatchNormalization(name='bn0', **bn_params)(x)
    # x = Activation('relu', name='relu0')(x)

    if block_type == 'transpose':
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                               for l in skip_connection_layers])

    cache = []
    for i in range(n_upsample_blocks):

        # check if there is a skip connection
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(decoder_filters[i], i, upsample_rate=upsample_rate,
                     skip=skip_connection, use_batchnorm=use_batchnorm)(x)
        cache.append(x)

    # TODO: hyper column
    for i in range(1, n_upsample_blocks):
        mid = UpSampling2D((2**i, 2**i))(cache[n_upsample_blocks-1-i])

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = mid._keras_shape[channel_axis]

        cache[n_upsample_blocks - 1 - i] = Conv2D(filters // 16, (1, 1))(mid)

        print(cache[n_upsample_blocks-1-i]._keras_shape)
    x = concatenate([l for l in cache])
    x = Dropout(0.5)(x)
    hypercolumn = Conv2D(classes, (3,3), padding='same', name='final_conv')(x)

    x = concatenate([hypercolumn, up_image_pool])
    x = Conv2D(classes, (3, 3), padding='same', name='final_final_conv')(x)

    x = Activation(activation, name=activation)(x)

    model = Model(input, [classify, hypercolumn, x])

    return model
