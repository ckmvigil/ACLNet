# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")

from sklearn.model_selection import train_test_split
import albumentations as A
import tensorflow as tf
from config import *
from utils import *
import numpy as np
import cv2
import os

train_augment = A.Compose([ 
    A.RandomCrop(height = CROP_SIZE[0], width = CROP_SIZE[1], p = 1), 
    A.CLAHE(p = 0.5),   
    A.RandomGamma(p = 0.5),
    A.OneOf([
            A.VerticalFlip(p = 0.25),
            A.HorizontalFlip(p = 0.25),
            A.Transpose(p = 0.25),
            A.RandomRotate90(p = 0.25),
        ], p = 1.0),
    A.OneOf([
             A.GridDistortion(p = 0.3),
             A.OpticalDistortion(distort_limit = 2, shift_limit = 0.5, p = 0.3),
             A.ElasticTransform(p = 0.3, alpha = 120, sigma = 120 * 0.05, alpha_affine = 120 * 0.03),
        ], p = 1.0),
    A.OneOf([
             A.RandomBrightness(p = 0.25),
             A.RandomContrast(p = 0.25),
    ], p = 1.0),
])

test_augment = A.RandomCrop(height = CROP_SIZE[0], width = CROP_SIZE[1], p = 1.0)

class ACLNetDataloader(tf.keras.utils.Sequence):
    """ Dataloader class to iterate over the data for 
       segmentation"""
    def __init__(self, batch_size, resize_size, crop_size, input_img_paths, target_img_paths, data_type):
        self.batch_size = batch_size
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.data_type = data_type
        self.train_transform = train_augment
        self.test_transform = test_augment

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size
    
    def rgbToOnehot(self, rgb_mat, color_dict = COLOR_VALUES):
        num_classes = len(color_dict)
        shape = rgb_mat.shape[:2]+(num_classes,)
        mat = np.zeros( shape, dtype=np.float32)
        for i, _ in enumerate(color_dict):
            mat[:, :, i] = np.all(rgb_mat.reshape((-1, 3)) == color_dict[i], axis = 1).reshape(shape[:2])
        return mat
    
    def KNNOutput(self, image):
        Z = image.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        K = 2
        _, label, center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result = res.reshape((image.shape))
        return result

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.crop_size + (3,), dtype = "float32")
        y = np.zeros((self.batch_size,) + self.crop_size + (2,), dtype = "float32")
        z = np.zeros((self.batch_size,) + self.crop_size + (3,), dtype = "float32")
        
        for j, (input_image, input_mask) in enumerate(zip(batch_input_img_paths, batch_target_img_paths)):
            image = cv2.imread(input_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.resize_size)
            
            mask = cv2.imread(input_mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, self.resize_size)
            
            if self.data_type == "Train":
                augment = self.train_transform(image = image, mask = mask)
            else:
                augment = self.test_transform(image = image, mask = mask)
            image = augment['image']
            mask = augment['mask']

            knn_image = self.KNNOutput(image)
            knn_image = knn_image.astype("float32")/255
            mask = self.rgbToOnehot(mask).astype('float32')
            mask = mask.astype('float32')
            x[j] = image.astype('float32')
            y[j] = mask
            z[j] = knn_image
            
        return (x, z), y

def getDataLoader(batch_size):
    """ Create dataloader and return dataloader object which can be used with 
        model.fit
    """
    input_img_paths = sorted([os.path.join(IMAGES_DIR, x) for x in os.listdir(IMAGES_DIR)])
    target_img_paths = sorted([os.path.join(MASKS_DIR, x) for x in os.listdir(MASKS_DIR)])

    X_train, X_test, y_train, y_test = train_test_split(input_img_paths, target_img_paths, test_size = TEST_SIZE, random_state = SEED)

    trainGen = ACLNetDataloader(batch_size = batch_size, resize_size = RESIZE_SIZE, crop_size = CROP_SIZE, input_img_paths = X_train, target_img_paths = y_train, data_type = "Train")
    testGen = ACLNetDataloader(batch_size = batch_size, resize_size = RESIZE_SIZE, crop_size = CROP_SIZE, input_img_paths = X_test, target_img_paths = y_test, data_type = "Test")

    return trainGen, testGen