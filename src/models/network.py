# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")

import tensorflow as tf
from config import *

def convBlock(
    block_input,
    num_filters = 256,
    kernel_size = 3,
    dilation_rate = 1,
    padding = "same",
    use_bias = False,
):
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size = kernel_size,
        dilation_rate = dilation_rate,
        padding = "same",
        use_bias = use_bias,
        kernel_initializer = tf.keras.initializers.HeNormal(),
    )(block_input)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size = (dims[-3], dims[-2]))(dspp_input)
    x = convBlock(x, kernel_size = 1, use_bias = True)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation = "bilinear",
    )(x)

    out_1 = convBlock(dspp_input, kernel_size = 1, dilation_rate = 1)
    out_6 = convBlock(dspp_input, kernel_size = 3, dilation_rate = 6)
    out_12 = convBlock(dspp_input, kernel_size = 3, dilation_rate = 12)
    out_18 = convBlock(dspp_input, kernel_size = 3, dilation_rate = 18)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convBlock(x, kernel_size = 1)
    return output

def getModel():
    """ ACLNet network
    """
    input_1 = tf.keras.layers.Input(shape = CROP_SIZE + (3,))
    input_2 = tf.keras.layers.Input(shape = CROP_SIZE + (3,))
    backbone = tf.keras.applications.EfficientNetB0(
        weights = "imagenet", include_top = False, input_tensor = input_1
    )
    
    x = backbone.get_layer("block6a_expand_activation").output
    x = DilatedSpatialPyramidPooling(x)

    attn = tf.keras.layers.Resizing(18, 18)(input_1)
    attn = tf.keras.layers.Conv2D(256, 1, 1, 'same', use_bias=False)(attn)
    attn = tf.keras.layers.Activation('softmax')(attn)

    x = tf.keras.layers.Multiply()([x, attn])
    input_a = tf.keras.layers.UpSampling2D(
        size = (CROP_SIZE[0] // 4 // x.shape[1], CROP_SIZE[1] // 4 // x.shape[2]),
        interpolation = "bilinear",
    )(x)
    input_b = backbone.get_layer("block3a_expand_activation").output
    input_b = convBlock(input_b, num_filters = 48, kernel_size = 1)

    knn_image = convBlock(input_2, kernel_size = 1)

    x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = convBlock(x)
    x = convBlock(x)
    x = tf.keras.layers.UpSampling2D(
        size = (CROP_SIZE[0] // x.shape[1], CROP_SIZE[1] // x.shape[2]),
        interpolation = "bilinear",
    )(x)
    x += knn_image 
    x = tf.keras.layers.Conv2D(NUM_CLASSES, kernel_size = (1, 1), padding = "same")(x)
    out = tf.keras.layers.Activation('sigmoid', name='segmentation')(x)

    model = tf.keras.models.Model(inputs = [input_1, input_2], outputs = [out])    
    return model