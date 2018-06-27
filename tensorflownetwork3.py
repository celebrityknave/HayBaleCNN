import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm      
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.activations import softmax
from tflearn.layers.estimator import regression

HAY_DIR = 'Subset/YesHayResized'
NOTHAY_DIR = 'Subset/NotHayResized'
IMG_SIZE = 256
BATCH_SIZE = 50
LR = 1e-6
MODEL_NAME = 'hay-convnet'
epochs = 250
save_dir = os.path.join(os.getcwd(),'saved_models')

model_name = 'tf_hayClassifierYOLO_model.h5'

def create_train_data():
    x = []
    for img in tqdm(os.listdir(HAY_DIR)):
        path = os.path.join(HAY_DIR, img)
        img_data = cv2.imread(path, 1)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data_flipped = cv2.flip(img_data, 1)
        x.append([np.array(img_data), np.array([1,0])])
        x.append([np.array(img_data_flipped), np.array([1,0])])

    for img in tqdm(os.listdir(NOTHAY_DIR)):
        path = os.path.join(NOTHAY_DIR, img)
        img_data = cv2.imread(path, 1)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        img_data_flipped = cv2.flip(img_data, 1)
        x.append([np.array(img_data), np.array([0,1])])
        x.append([np.array(img_data_flipped), np.array([0,1])])

    shuffle(x)
    testSetSize = np.floor_divide(len(x),10)
    testing_data = x[0:testSetSize]
    training_data = x[testSetSize:]

    np.save('train_data.npy', training_data)
    np.save('test_data.npy',testing_data)
    return training_data,testing_data

# If dataset is not created:
train_data,test_data = create_train_data()
# If you have already created the dataset:
#train_data = np.load('train_data.npy')
#test_data = np.load('test_data.npy')

X_train = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = [i[1] for i in train_data]
X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_test = [i[1] for i in test_data]

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
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=epochs, 
          validation_set=({'input': X_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, batch_size=BATCH_SIZE, run_id=MODEL_NAME)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir,model_name)
model.save(model_path)
print('Saved trained model at %s' % model_path)

"""
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(LR))
plt.show()
"""
