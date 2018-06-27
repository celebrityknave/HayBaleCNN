import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm      
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#HAY_DIR = 'Subset/YesHayResized'
#NOTHAY_DIR = 'Subset/NotHayResized'
IMG_SIZE = 128
LR = 1e-6
MODEL_NAME = 'hay-convnet'
epochs = 250
save_dir = os.path.join(os.getcwd(),'saved_models')
model_dir= 'saved_models/tf_hayClassifierYOLO_model.h5'

#x_placeholder = tf.placeholder(tf.float32, [1,128,128,3])

def load_image(img,hayDir):
    #img = '4.jpg'
    path = os.path.join(hayDir,img)
    print(path)
    img_data = cv2.imread(path,1)
    img_data = cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))

    display_image(img_data)
    return img_data

def display_image(img_data):
    #Display sample picture
    fig = plt.figure(figsize=(2,2))
    plt.imshow(img_data)
    plt.show()

def build_network():
        tf.reset_default_graph()
    convnet = input_data(shape=[IMG_SIZE, IMG_SIZE, 3], name='input')
    convnet = conv_2d(convnet, 32, 3, activation='relu',regularizer='L2')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 64, 3, activation='relu',regularizer='L2')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 3, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet, 64, 1, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet, 128, 3, activation='relu',regularizer='L2')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 3, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet, 128, 1, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet, 256, 3, activation='relu',regularizer='L2')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet,512,3, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet,256,1, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet,512,3, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet,256,1, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet,512,3, activation='relu',regularizer='L2')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet,1024,3, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet,512,1, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet,1024,3, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet,512,1, activation='relu',regularizer='L2')
    convnet = conv_2d(convnet,1024,3, activation='relu',regularizer='L2')

    convnet = dropout(convnet,0.7)
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    return convnet

model = tflearn.DNN(build_network(), tensorboard_dir='log', tensorboard_verbose=0)
model.load(model_dir)

def predimg(imgdir,hayDir):
    x = load_image(imgdir,hayDir)
    x = np.expand_dims(x,0)

    prediction = model.predict(x)
    prediction = np.rint(prediction)
    print(prediction)
