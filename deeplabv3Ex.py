#tensorflow 2.5.0

# importing all packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ZeroPadding2D, Dense


def changePadding(x, stride, kernel_size, rate):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    return x, depth_padding

# defining the seperable convolution

def SepConv(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

# defining the xception model function

def xception_model(img_input):

    x_conv1 = Conv2D(32, (3, 3), strides=(2, 2), name='entry_flow_conv1_1', use_bias=False, padding='same', dilation_rate=(1, 1))(img_input)
    x_conv1 = BatchNormalization(name='entry_flow_conv1_1_BN')(x_conv1)
    x_conv1 = Activation('relu')(x_conv1)

    x_conv1 = Conv2D(64, (3, 3), strides=(1, 1), use_bias=False, padding='same', name ='entry_flow_conv1_2', dilation_rate=(1, 1))(x_conv1)
    x_conv1 = BatchNormalization(name='entry_flow_conv1_2_BN')(x_conv1)
    x_conv1 = Activation('relu')(x_conv1)

    x_conv_str, depth_padding = changePadding(x_conv1, 2, 1, 1)
    x_conv_str = Conv2D(128, (1, 1), strides=(2, 2), use_bias=False, padding='valid', name='entry_flow_block1_shortcut', dilation_rate=(1, 1))(x_conv_str)

    x_sep1 = SepConv(x_conv1, 128, 'entry_flow_block1_separable_conv{}'.format(1), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
    x_sep1 = SepConv(x_sep1, 128, 'entry_flow_block1_separable_conv{}'.format(2), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
    x_sep1 = SepConv(x_sep1, 128, 'entry_flow_block1_separable_conv{}'.format(3), stride=2, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)

    x_conv2 = tf.keras.layers.add([x_sep1, x_conv_str])
   
    x_conv_str, depth_padding = changePadding(x_conv2, 2, 1, 1)
    x_conv_str = Conv2D(256, (1, 1), strides=(2, 2), use_bias=False, padding='valid', name = 'entry_flow_block2_shortcut', dilation_rate=(1, 1))(x_conv_str)
    x_conv_str = BatchNormalization(name='entry_flow_block2_shortcut_BN')(x_conv_str)

    x_sep2 = SepConv(x_conv2, 256, 'entry_flow_block2_separable_conv{}'.format(1), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
    x_sep2 = SepConv(x_sep2, 256, 'entry_flow_block2_separable_conv{}'.format(2), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
    skip1 = x_sep1
    x_sep2 = SepConv(x_sep2, 256, 'entry_flow_block2_separable_conv{}'.format(3), stride=2, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)

    x_conv3 = tf.keras.layers.add([x_sep2, x_conv_str])
    
    x_conv_str, depth_padding = changePadding(x_conv3, 2, 1, 1)
    x_conv_str = Conv2D(728, (1, 1), strides=(2, 2), use_bias=False, padding='valid', name = 'entry_flow_block3_shortcut', dilation_rate=(1, 1))(x_conv_str)
    x_conv_str = BatchNormalization(name='entry_flow_block3_shortcut_BN')(x_conv_str)

    x_sep3 = SepConv(x_conv3, 728, 'entry_flow_block3_separable_conv{}'.format(1), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
    x_sep3 = SepConv(x_sep3, 728, 'entry_flow_block3_separable_conv{}'.format(2), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
    x_sep3 = SepConv(x_sep3, 728, 'entry_flow_block3_separable_conv{}'.format(3), stride=2, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
    
    x_conv4 = tf.keras.layers.add([x_sep3, x_conv_str])

    for i in range(16):
        x_sep4 = SepConv(x_conv4, 728, 'middle_flow_unit_{}'.format(i + 1)+'_separable_conv{}'.format(1), stride=1, kernel_size=3, rate=1,     depth_activation=False, epsilon=1e-3)
        x_sep4 = SepConv(x_sep4, 728, 'middle_flow_unit_{}'.format(i + 1)+'_separable_conv{}'.format(2), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
        x_sep4 = SepConv(x_sep4, 728, 'middle_flow_unit_{}'.format(i + 1)+'_separable_conv{}'.format(3), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
        x_conv4 = tf.keras.layers.add([x_sep4, x_conv4])

    x_conv_str = Conv2D(1024, (1, 1), strides=(1, 1), use_bias=False, padding='same', name = 'exit_flow_block1_shortcut', dilation_rate=(1, 1))(x_conv4)
    x_conv_str = BatchNormalization(name='exit_flow_block1_shortcut_BN')(x_conv_str)

    x_sep5 = SepConv(x_conv4, 728, 'exit_flow_block1_separable_conv{}'.format(1), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
    x_sep5 = SepConv(x_sep5, 1024, 'exit_flow_block1_separable_conv{}'.format(2), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)
    x_sep5 = SepConv(x_sep5, 1024, 'exit_flow_block1_separable_conv{}'.format(3), stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3)

    x_conv5 = tf.keras.layers.add([x_sep5, x_conv_str])

    x_sep6 = SepConv(x_conv5, 1536, 'exit_flow_block2_separable_conv{}'.format(1), stride=1, kernel_size=3, rate=2, depth_activation=True, epsilon=1e-3)
    x_sep6 = SepConv(x_sep6, 1536, 'exit_flow_block2_separable_conv{}'.format(2), stride=1, kernel_size=3, rate=2, depth_activation=True, epsilon=1e-3)
    x_sep6 = SepConv(x_sep6, 2048, 'exit_flow_block2_separable_conv{}'.format(3), stride=1, kernel_size=3, rate=2, depth_activation=True, epsilon=1e-3)

    return x_sep6, skip1

# defining the aspp function

def aspp(input):
    
    x = input
    atrous_rates = (1, 2, 3)
    shape_before = tf.shape(x)

    b4 = GlobalAveragePooling2D()(x)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    
    size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3], method='bilinear', align_corners=True))(b4)
  
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    b1 = SepConv(x, 256, 'aspp1', rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    b2 = SepConv(x, 256, 'aspp2', rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    b3 = SepConv(x, 256, 'aspp3', rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    x = Concatenate()([b4, b0, b1, b2, b3])
    return x;

# defining the decoder function

def decoder(img_input, x, skip1, classes):

    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, skip1.shape[1:3], method='bilinear', align_corners=True))(x)

    dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)

    x = Concatenate()([x, dec_skip1])
    x = SepConv(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
    x = SepConv(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

    x = Conv2D(classes, (1, 1), padding='same', name='logits_semantic')(x)
    x=  Dense(1,activation='sigmoid')(x)

    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, size_before3[1:3], method='bilinear', align_corners=True))(x)
    
    return x;

# defining the deeplab V3+ model function

def Deeplabv3(weights='None',input_shape=(100,100,3), classes=2, OS=16):


    img_input = Input(shape=input_shape) 

    x, skip1 = xception_model(img_input)
    
    x = aspp(x)

    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    
    x = decoder(img_input, x, skip1, classes)
    

    inputs = img_input
    print(str(input_shape[0])+','+str(input_shape[1]))
    model = Model(inputs, x, name='deeplabv3plus')
    return model
