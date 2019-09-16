import pandas as pd
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
import pydot, graphviz
from pathlib import Path
import json
import itertools
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import sys
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from evaluationmatrix import fpr, weighted_average_recall
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import confusion_matrix
import scipy.io as sio

from keras.models import Sequential, Model
from keras.utils import np_utils, plot_model, Sequence
from keras import metrics
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import LSTM, GlobalAveragePooling2D, GRU, Bidirectional, UpSampling2D
from keras.layers import BatchNormalization, Input, Activation, Lambda, concatenate, add
from keras.engine import InputLayer
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
	splittensor, Softmax4D
from theano import tensor as T
from keras.layers import Multiply, Concatenate, Add


# from utilities import loading_smic_table, loading_samm_table, loading_casme_table
# from utilities import class_merging, read_image, create_generator_LOSO
# from utilities import LossHistory, record_loss_accuracy
# from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from models import VGG_16, temporal_module, layer_wise_conv_autoencoder, layer_wise_autoencoder, convolutional_autoencoder, alexnet
from models import tensor_reshape, l2_normalize, l2_normalize_output_shape, repeat_element_autofeat





def train_res50_imagenet(classes = 5, freeze_flag = 'last'):
	resnet50 = ResNet50(weights = 'imagenet')

	# # load macro
	# last_layer = resnet50.layers[-2].output
	# dense_classifier = Dense(6, activation = 'softmax')(last_layer)
	# resnet50 = Model(inputs = resnet50.input, outputs = dense_classifier)		
	# resnet50.load_weights('res_micro_grayscale_augmentedres50_retrain.h5')	

	last_layer = resnet50.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	resnet50 = Model(inputs = resnet50.input, outputs = dense_classifier)
	plot_model(resnet50, to_file='resnet50.png', show_shapes=True)

	for layer in resnet50.layers:
		layer.trainable = True

	# # for 2nd last block
	if freeze_flag == '2nd_last':
		for layer in resnet50.layers[:-25]:
			layer.trainable = False	

	# # for 3rd last block
	elif freeze_flag == '3rd_last':
		for layer in resnet50.layers[:-37]:
			layer.trainable = False		

	# # last
	elif freeze_flag == 'last':
		for layer in resnet50.layers[:-14]:
			layer.trainable = False	
	print(resnet50.summary())

	return resnet50

def train_vgg16_imagenet(classes = 5, freeze_flag = 'last'):
	vgg16 = VGG16(weights = 'imagenet')


	last_layer = vgg16.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	vgg16 = Model(inputs = vgg16.input, outputs = dense_classifier)	
	plot_model(vgg16, to_file='vgg16.png', show_shapes=True)
		
	# for layer in vgg16.layers:
	# 	layer.trainable = True

	# train last 2 block
	if freeze_flag == '2nd_last':
		for layer in vgg16.layers[:-8]:
			layer.trainable = False

	# # train last 3 block
	elif freeze_flag == '3rd_last':
		for layer in vgg16.layers[:-9]:
			layer.trainable = False

	# # train last block
	elif freeze_flag == 'last':
		for layer in vgg16.layers[:-7]:
			layer.trainable = False	
	print(vgg16.summary())
	return vgg16

def train_inceptionv3_imagenet(classes = 5, freeze_flag = 'last'):
	inceptionv3 = InceptionV3(weights = 'imagenet')



	last_layer = inceptionv3.layers[-2].output
	dense_classifier = Dense(classes, activation = 'softmax')(last_layer)
	inceptionv3 = Model(inputs = inceptionv3.input, outputs = dense_classifier)	
	plot_model(inceptionv3, to_file='inceptionv3.png', show_shapes=True)

	for layer in inceptionv3.layers:
		layer.trainable = True
	
	# # 2nd last incep block
	if freeze_flag == '2nd_last':
		for layer in inceptionv3.layers[:-85]:
			layer.trainable = False

	# # 3rd last incep block
	elif freeze_flag == '3rd_last':
		for layer in inceptionv3.layers[:-117]:
			layer.trainable = False

	# # last block
	elif freeze_flag == 'last':
		for layer in inceptionv3.layers[:-34]:
			layer.trainable = False	

	print(inceptionv3.summary())

	return inceptionv3

def train_xception_imagenet():
	xception = Xception(weights = 'imagenet')
	last_layer = xception.layers[-2].output
	dense_classifier = Dense(5, activation = 'softmax')(last_layer)
	xception = Model(inputs = xception.input, outputs = dense_classifier)	
	plot_model(xception, to_file='xception.png', show_shapes=True)

	return xception

def test_alexnet_imagenet(classes = 5):
	model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	model.load_weights('alexnet_weights.h5')

	model = Model(inputs = model.input, outputs = model.layers[-2].output)
	plot_model(model, to_file = 'alexnet', show_shapes = True)
	print(model.summary())

	return model	

def train_alexnet_imagenet(classes = 5):
	model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	model.load_weights('alexnet_weights.h5')

	# add in own classes ( maybe not necessary)
	last_layer = model.layers[-3].output
	dense_classifier = Dense(5, activation = 'softmax', name='me_dense')(last_layer)
	model = Model(inputs = model.input, outputs = dense_classifier)
	plot_model(model, to_file = 'alexnet', show_shapes = True)
	print(model.summary())

	# freezing

	# # 3rd Last
	# for layer in model.layers[:-22]:
	# 	layer.trainable = False

	# # 2nd Last
	# for layer in model.layers[:-20]:
	# 	layer.trainable = False

	# # Last
	# for layer in model.layers[:-14]:
	# 	layer.trainable = False	

	return model

def train_shallow_alexnet_imagenet(classes = 5, freeze_flag = None):
	model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	model.load_weights('alexnet_weights.h5')

	# modify architecture
	# ##################### Ori #####################
	# last_conv_1 = model.layers[5].output
	# conv_2 = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv_2', kernel_initializer='he_normal', bias_initializer='he_normal')(last_conv_1)
	# conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	# conv_2 = crosschannelnormalization(name="convpool_2")(conv_2)
	# conv_2 = ZeroPadding2D((2,2))(conv_2)

	# conv_2 = Flatten(name="flatten")(conv_2)
	# conv_2 = Dropout(0.5)(conv_2)
	# ##############################################

	################# Use 2 conv with weights ######################
	conv_2 = model.layers[13].output
	conv_2 = Flatten(name = 'flatten')(conv_2)
	conv_2 = Dropout(0.5)(conv_2)
	################################################################

	# ##### FC for experiments #####
	# fc_1 = Dense(4096, activation='relu', name='fc_1', kernel_initializer='he_normal', bias_initializer='he_normal')(conv_2)
	# fc_2 = Dense(4096, activation='relu', name='fc_2', kernel_initializer='he_normal', bias_initializer='he_normal')(fc_1)
	# ##############################

	##### GAP experiments #####
	# gap = GlobalAveragePooling2D(data_format = 'channels_first')(conv_2)
	# gap = Dropout(0.5)(gap)
	###########################

	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(conv_2)
	prediction = Activation("softmax")(dense_1)

	model = Model(inputs = model.input, outputs = prediction)		
	plot_model(model, to_file='shallowalex', show_shapes =True)
	print(model.summary())
	return model

def train_shallow_alexnet_imagenet_FCN(classes = 5, freeze_flag = None):
	model = alexnet(input_shape = (3, 227, 227), nb_classes = 1000, mean_flag = True)
	model.load_weights('alexnet_weights.h5')

	# modify architecture
	last_conv_1 = model.layers[5].output
	conv_2 = Conv2D(256, (5, 5), strides=(1, 1), activation='relu', name='conv_2', kernel_initializer='he_normal', bias_initializer='he_normal')(last_conv_1)
	conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_2 = crosschannelnormalization(name="convpool_2")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)

	conv_2 = Dropout(0.5)(conv_2)
	
	conv_activate = Conv2D(classes, kernel_size=(1, 1), strides = (1, 1), activation = 'relu', kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='conv_activate')(conv_2)	
	conv_activate = GlobalAveragePooling2D(data_format = 'channels_first')(conv_activate)

	model = Model(inputs = model.input, outputs = conv_activate)
	plot_model(model, to_file='shallowalex_fcn', show_shapes =True)
	print(model.summary())

	return model	



def train_dual_stream_shallow_alexnet(classes = 5, freeze_flag=None):
	sys.setrecursionlimit(10000)

	input_mag = Input(shape=(3, 227, 227))
	input_strain = Input(shape=(3, 227, 227))
	model_mag = train_shallow_alexnet_imagenet(classes = classes)
	model_strain = train_shallow_alexnet_imagenet(classes = classes)

	# FOR MULTIPLYING / ADDITION
	model_mag = Model(inputs = model_mag.input, outputs = model_mag.layers[-5].output)
	model_strain = Model(inputs = model_strain.input, outputs = model_strain.layers[-5].output)

	plot_model(model_mag, show_shapes=True, to_file = 'model_mag')
	plot_model(model_strain, show_shapes=True, to_file = 'model_strain')

	# # FOR CONCATENATION
	# model_mag = Model(inputs = model_mag.input, outputs = model_mag.layers[-4].output)
	# model_strain = Model(inputs = model_strain.input, outputs = model_strain.layers[-4].output)

	flatten_mag = model_mag(input_mag)
	flatten_strain = model_strain(input_strain)

	plot_model(model_mag, to_file = 'mag_model', show_shapes=True)
	plot_model(model_strain, to_file = 'strain_model', show_shapes=True)

	# concatenate FOR MULTIPLY OR ADD
	concat = Multiply()([flatten_mag, flatten_strain])
	# concat = Add()([flatten_mag, flatten_strain])

	# # # concatenate FOR CONCATENATION
	# concat = Concatenate(axis=1)([flatten_mag, flatten_strain])



	concat = Flatten()(concat) # FOR MULTIPLY ADD
	#concat = Lambda(l2_normalize, output_shape=l2_normalize_output_shape)(concat)
	dropout = Dropout(0.5)(concat)

	# fc_1 = Dense(4096, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(dropout)
	# fc_2 = Dense(4096, kernel_initializer = 'he_normal', bias_initializer = 'he_normal')(fc_1)

	dense_1 = Dense(classes, kernel_initializer = 'he_normal', bias_initializer = 'he_normal', name='last_fc')(dropout)
	prediction = Activation("softmax", name = 'softmax_activate')(dense_1)


	model = Model(inputs = [input_mag, input_strain], outputs = prediction)
	plot_model(model, to_file = 'train_dual_stream_shallow_alexnet', show_shapes=True)
	print(model.summary())

	return model

