# -*- coding: utf-8 -*-
import tensorflow as tf

def diceCoef(y_true, y_pred, smooth=tf.keras.backend.epsilon()):   
    y_true_f = tf.keras.backend.flatten(y_true)    
    y_pred_f = tf.keras.backend.flatten(y_pred)    
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)    
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f * y_true_f) + tf.keras.backend.sum(y_pred_f * y_pred_f) + smooth)

def diceCoefLoss(y_true, y_pred):
    return 1.0 - diceCoef(y_true, y_pred)

def bceDiceLoss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + diceCoefLoss(y_true, y_pred)
    return loss
