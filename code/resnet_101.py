# -*- coding: utf-8 -*-

from keras.optimizers import SGD, RMSprop #, RMSpropAccum
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, Dropout, GlobalAveragePooling2D, UpSampling2D, Conv2DTranspose, LeakyReLU
from keras.layers.merge import add, concatenate, average
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

from custom_layers.scale_layer import Scale
# from loss import focal_loss
from constants import *

import sys
sys.setrecursionlimit(3000)

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False, trainable=trainable)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a', trainable=trainable)(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a', trainable=trainable)(x)
    x = Activation('relu', name=conv_name_base + '2a_relu', trainable=trainable)(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding', trainable=trainable)(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b', trainable=trainable)(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b', trainable=trainable)(x)
    x = Activation('relu', name=conv_name_base + '2b_relu', trainable=trainable)(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c', trainable=trainable)(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c', trainable=trainable)(x)

    x = add([x, input_tensor], name='res' + str(stage) + block, trainable=trainable)
    x = Activation('relu', name='res' + str(stage) + block + '_relu', trainable=trainable)(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, trainable, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False, trainable=trainable)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a', trainable=trainable)(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a', trainable=trainable)(x)
    x = Activation('relu', name=conv_name_base + '2a_relu', trainable=trainable)(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding', trainable=trainable)(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b', trainable=trainable)(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b', trainable=trainable)(x)
    x = Activation('relu', name=conv_name_base + '2b_relu', trainable=trainable)(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c', trainable=trainable)(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c', trainable=trainable)(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False, trainable=trainable)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1', trainable=trainable)(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1', trainable=trainable)(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block, trainable=trainable)
    x = Activation('relu', name='res' + str(stage) + block + '_relu', trainable=trainable)(x)
    return x

def resnet101_model(img_rows, img_cols, color_type=1, trainable=True):
    """
    Resnet 101 Model for Keras

    Model Schema and layer naming follow that of the original Caffe implementation
    https://github.com/KaimingHe/deep-residual-networks

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding', trainable=trainable)(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1', trainable=trainable)(x)
    x = Scale(axis=bn_axis, name='scale_conv1', trainable=trainable)(x)
    x = Activation('relu', name='conv1_relu', trainable=trainable)(x)

    layer1 = x

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1', trainable=trainable)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

    layer2 = x
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
    for i in range(1,3):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i), trainable=trainable)

    layer3 = x
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
    for i in range(1,23):
      # if i == 22:
      #   x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i), trainable=True)
      # else:
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i), trainable=trainable)

    layer4 = x
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', trainable=trainable)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)

    # x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    # x_fc = Flatten()(x_fc)
    # x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # model = Model(img_input, x_fc)

    model = Model(img_input, x)

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = '../imagenet_models/resnet101_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = '../imagenet_models/resnet101_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    return model, [layer1, layer2, layer3, layer4]


def upsample(in_layer, down, nchan):
    up = Conv2D(nchan, (1, 1), strides=(1, 1), kernel_initializer='he_uniform')(in_layer)
    up = Conv2DTranspose(nchan, (3, 3), strides=(2, 2), padding="same")(up)
    down = Conv2D(nchan, (1, 1), strides=(1, 1), kernel_initializer='he_uniform')(down)
    # up = UpSampling2D((2, 2))(up)
    up = concatenate([down, up], axis=3)
    # up = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(up)
    # up = BatchNormalization()(up)
    # up = Activation('relu')(up)
    # up = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(up)
    # up = BatchNormalization()(up)
    # up = Activation('relu')(up)
    # up = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(up)
    # up = BatchNormalization()(up)
    up = Activation('relu')(up)
    return up

def unet_resnet101(img_rows, img_cols, color_type, num_classes=1):
    encode_model, layers = resnet101_model(img_rows, img_cols, color_type)

    input = encode_model.input
    # layer1 = encode_model.get_layer('conv1_relu')
    # layer2 = encode_model.get_layer('res2c_relu')
    # layer3 = encode_model.get_layer('res3b2_relu')
    # layer4 = encode_model.get_layer('res4b22_relu')

    layer1, layer2, layer3, layer4 = layers

    x = encode_model.output
    x = upsample(x, layer4, 1024 // 2)
    x = upsample(x, layer3, 512 // 2)
    x = upsample(x, layer2, 256 // 2)
    x = upsample(x, layer1, 128 // 2)
    x = upsample(x, input, 64 // 2)

    output1 = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    if not USE_REFINE_NET:
        model = Model(inputs=input, outputs=[output1])

        # model.load_weights('../weights/head-segmentation-model.h5')
        # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])

        return model

    else:
        inputs2 = concatenate([input, output1])
        outputs2 = block2(inputs2, 64)
        outputs2 = average([output1, outputs2])
        model2 = Model(inputs=input, outputs=[output1, outputs2])
        return model2


def block2(in_layer, nchan, num_classes=1, relu=False):
    b1 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(in_layer)
    # b1 = BatchNormalization()(b1)
    if relu:
        b1 = Activation('relu')(b1)
    else:
        b1 = LeakyReLU(0.0001)(b1)

    b2 = Conv2D(nchan, (3, 3), padding='same')(b1)
    # b2 = BatchNormalization()(b2)
    if relu:
        b2 = Activation('relu')(b2)
    else:
        b2 = LeakyReLU(0.0001)(b2)

    b3 = Conv2D(nchan, (3, 3), padding='same')(b2)
    # b3 = BatchNormalization()(b3)
    if relu:
        b3 = Activation('relu')(b3)
    else:
        b3 = LeakyReLU(0.0001)(b3)

    b4 = Conv2D(num_classes, (3, 3), padding='same', activation='sigmoid')(b3)
    # b4 = BatchNormalization()(b4)
    # b4 = Activation('sigmoid')(b4)
    return b4
