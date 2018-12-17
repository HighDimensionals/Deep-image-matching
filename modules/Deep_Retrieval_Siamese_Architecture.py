
from __future__ import division
from __future__ import print_function

rmac_file_path = "../keras_rmac-master"
import sys
sys.path.append(rmac_file_path)

from keras.layers import Lambda, Dense, TimeDistributed, Input, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K

from keras.applications.vgg16 import VGG16
from RoiPooling import RoiPooling

import scipy.io
import numpy as np
import utils

from keras.layers import Concatenate

def addition(x):
    sum = K.sum(x, axis=1,keepdims=False)
    return sum


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, 512, axis=-1)
    out = x * w
    return out

# Class for triplet loss
class TripletLoss:
    def __init__(self,margin,batch_size):
        self.margin=margin
        self.batch_size=batch_size
        
    def triplet_loss_deep_retrieval(self,y_true,y_pred):
        loss_sum=0
        for i in range(self.batch_size):
            user_latent,positive_item_latent, negative_item_latent = y_pred[i,0,:],y_pred[i,1,:],y_pred[i,2,:]
            loss = K.maximum(K.constant(0), K.constant(self.margin) + 
                K.sum(K.square(user_latent - positive_item_latent), axis=-1, keepdims=True) -
                K.sum(K.square(user_latent - negative_item_latent), axis=-1, keepdims=True))
            loss_sum+=loss

        return loss_sum


def deep_retrieval_siamese(input_shape, model_type):

    # Load VGG16 model
    img_input = Input(shape=input_shape)
    vgg16_model =VGG16(include_top=False, weights='imagenet', input_tensor=img_input, input_shape=input_shape, pooling=None, classes=1000)
    
    #Inputs
    img_input_1 = Input(shape=input_shape)
    img_input_2 = Input(shape=input_shape)
    img_input_3 = Input(shape=input_shape)
    
    #==========VGG16 model outputs=================
    vgg16_output_1 = vgg16_model(img_input_1)
    vgg16_output_2 = vgg16_model(img_input_2)
    vgg16_output_3 = vgg16_model(img_input_3)
    
   
    #=================Max/Sum Pooling================
    if model_type is 'mac':
        pool_model = MaxPooling2D(pool_size=(7, 7), strides=None, padding='valid')
    elif model_type is 'spoc':
        pool_model = AveragePooling2D(pool_size=(7, 7), strides=None, padding='valid')
      
    Pooling_output_1 = Lambda(lambda x: K.squeeze(K.squeeze(pool_model(x),axis=1),axis=1),name='query_pooling')(vgg16_output_1)
    Pooling_output_2 = Lambda(lambda x: K.squeeze(K.squeeze(pool_model(x),axis=1),axis=1),name='relevant_pooling')(vgg16_output_2)
    Pooling_output_3 = Lambda(lambda x: K.squeeze(K.squeeze(pool_model(x),axis=1),axis=1),name='irrelevant_pooling')(vgg16_output_3)
    
    #=================Sum Pooling================
    if model_type is 'spoc':
        pool_model = AveragePooling2D(pool_size=(7, 7), strides=None, padding='valid')
      
        Pooling_output_1 = Lambda(lambda x: K.squeeze(K.squeeze(pool_model(x),axis=1),axis=1),name='query_pooling')(vgg16_output_1)
        Pooling_output_2 = Lambda(lambda x: K.squeeze(K.squeeze(pool_model(x),axis=1),axis=1),name='relevant_pooling')(vgg16_output_2)
        Pooling_output_3 = Lambda(lambda x: K.squeeze(K.squeeze(pool_model(x),axis=1),axis=1),name='irrelevant_pooling')(vgg16_output_3)
    #=================Normalization================
    Norm1_output_1 = Lambda(lambda x: K.l2_normalize(x, axis=1), name='query_norm1')(Pooling_output_1)
    Norm1_output_2 = Lambda(lambda x: K.l2_normalize(x, axis=1), name='relevant_norm1')(Pooling_output_2)
    Norm1_output_3 = Lambda(lambda x: K.l2_normalize(x, axis=1), name='irrelevant_norm1')(Pooling_output_3)

    #===================PCA Layer=================
    in_pca = Input(shape=(1,512))
    out_pca = Dense(512, name='PCA',
                              kernel_initializer='identity',
                              bias_initializer='zeros')(in_pca)
    pca_model = Model(input=in_pca, output=out_pca)
    
    PCA_output_1 = pca_model(Norm1_output_1)
    PCA_output_2 = pca_model(Norm1_output_2)
    PCA_output_3 = pca_model(Norm1_output_3)
    
    #=================Normalization================
    Norm2_output_1 = Lambda(lambda x: K.l2_normalize(x, axis=1), name='query_norm2')(PCA_output_1)
    Norm2_output_2 = Lambda(lambda x: K.l2_normalize(x, axis=1), name='relevant_norm2')(PCA_output_2)
    Norm2_output_3 = Lambda(lambda x: K.l2_normalize(x, axis=1), name='irrelevant_norm2')(PCA_output_3)

    y_pred = Lambda(lambda x: K.concatenate([K.expand_dims(x[0],axis=1),K.expand_dims(x[1],axis=1),K.expand_dims(x[2],axis=1)],axis=1))([Norm2_output_1,Norm2_output_2,Norm2_output_3])
    
    # Define model
    model = Model(input=[img_input_1,img_input_2,img_input_3], 
                  output=[y_pred])#rmac_norm)

    return model


