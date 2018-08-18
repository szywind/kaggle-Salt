# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D, Dropout, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

from custom_layers.scale_layer import Scale
from constants import *
from keras.layers.merge import concatenate, average

def densenet161_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    '''
    DenseNet 161 Model for Keras

    Model Schema is based on
    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfVnlCMlBGTDR3RGs
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfUDZwVjU2cFNidTA

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    layers = []

    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(img_rows, img_cols, 3), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(3, img_rows, img_cols), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [6,12,36,24] # For DenseNet-161

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)

    layers.append(x)

    # x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding = 'same', name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        layers.append(x)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    # x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    # x_fc = Dense(1000, name='fc6')(x_fc)
    # x_fc = Activation('softmax', name='prob')(x_fc)

    # model = Model(img_input, x_fc, name='densenet')

    model = Model(img_input, x, name='densenet')

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = '../imagenet_models/densenet161_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = '../imagenet_models/densenet161_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    return model, layers


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

def upsample(in_layer, down, nchan, dropout_rate = 0.15):
    up = Conv2D(nchan, (1, 1), strides=(1, 1), kernel_initializer='he_uniform')(in_layer)
    up = BatchNormalization()(up)
    # up = Dropout(dropout_rate)(up)
    up = Activation('relu')(up)
    up = Conv2DTranspose(nchan, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(up)
    down = Conv2D(nchan, (1, 1), strides=(1, 1), kernel_initializer='he_uniform')(down)
    # up = UpSampling2D((2, 2))(up)

    up = concatenate([down, up], axis=3)
    # up = Dropout(dropout_rate)(up)
    up = Activation('relu')(up)

    up = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(up)
    up = BatchNormalization()(up)
    # up = Dropout(dropout_rate)(up)
    up = Activation('relu')(up)

    # up = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(up)
    # up = BatchNormalization()(up)
    # # up = Dropout(dropout_rate)(up)
    # up = Activation('relu')(up)

    # up = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(up)
    # up = BatchNormalization()(up)
    # up = Activation('relu')(up)
    return up

def unet_densenet161(img_rows, img_cols, color_type, num_classes=1):
    # encode_model, layers = densenet169_model(img_rows, img_cols, color_type, dropout_rate=0.15)
    encode_model, layers = densenet161_model(img_rows, img_cols, color_type)

    input = encode_model.input
    # layer1 = encode_model.get_layer('conv1_relu')
    # layer2 = encode_model.get_layer('res2c_relu')
    # layer3 = encode_model.get_layer('res3b2_relu')
    # layer4 = encode_model.get_layer('res4b22_relu')


    x = encode_model.output
    nchan = 512
    for layer in reversed(layers):
        x = upsample(x, layer, nchan)
        nchan  = nchan // 2
    x = upsample(x, input, nchan)

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