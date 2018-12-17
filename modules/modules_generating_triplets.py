
from PIL import Image
from pathlib import Path

import pandas as pd
import numpy as np
import csv
import h5py
import os

from keras.preprocessing import image
#=============================================================
## Generate feature vectors
# Inputs
# model: used for feature extraction
# input_csv_path: Path used for loading train data frame
# imagesDir: Directory of images
# features_path: Path where all the feature vectors are saved, along with the image names
def generate_feature_vectors(model,input_csv_path,imagesDir,features_path,len_features):
    print("Generating feature vectors...")
    # Load train data frame
    df=pd.read_csv(input_csv_path)
    batch_size=1
    ls_image_names = df["image_name"].tolist()
    # Image parameters
    img_height, img_width = 224,224
    channels=3

    features_unique_image_train = np.zeros((len(df), len_features))
    for i in range(len(df)):
        img = image.load_img(imagesDir+"/"+ls_image_names[i], target_size=(img_height, img_width))
        img = image.img_to_array(img)
        img = img/255
        features_unique_image_train[i]=model.predict([np.expand_dims(img,axis=0),np.expand_dims(img,axis=0),np.expand_dims(img,axis=0)])[0,0,:]
    print("Feature vectors genereated for all train vectors...")

    # Save train feature vectors
    h5f = h5py.File(features_path, 'w')
    h5f.create_dataset('dataset_1', data = features_unique_image_train)
    ls_image_names = [str.encode(name) for name in ls_image_names]
    h5f.create_dataset('dataset_2', data = ls_image_names)
    h5f.close()

#=============================================================
# ## Load training feature vectors
# features_h5 = h5py.File(features_path, 'r')
# unique_features = np.array(features_h5.get('dataset_1'))
# unique_filenames = list(features_h5.get('dataset_2'))
# unique_filenames=[filename.decode("utf-8") for filename in unique_filenames]
# features_h5.close()
#=============================================================
# Used to generate the feature vectors, during resampling of triplets
## Inputs
# input_csv_path: csv of train images
# new_csv_path: csv of train images used for generating triplet candidates
# imagesDir: directory of images
# features_path: file to which features are saved
# batch_size: batch_size of the model
def generate_feature_vectors_resampling(model,input_csv_path,imagesDir,features_path,len_features,batch_size=1):
    print("Generating feature vectors...")
    # Load train data frame
    df=pd.read_csv(input_csv_path)
    
    #Extract image names
    ls_image_names = df["image_name"].tolist()
    # Image parameters
    img_height, img_width = 224,224
    channels=3
    
    #Number of model prediction steps, with batch size of batch_size
    steps=int(len(df)/batch_size)
    features_unique_image_train = np.zeros((len(df), len_features))
    batch_img=np.zeros((batch_size,224,224,3))
    for i in range(steps):
        start_index=i*batch_size
        for j in range(batch_size):
            img = image.load_img(imagesDir+"/"+ls_image_names[start_index+j], target_size=(img_height, img_width))
            img = image.img_to_array(img)
            img = img/255.
            batch_img[j]=img
        features_unique_image_train[start_index:start_index+batch_size]=model.predict([batch_img,batch_img,batch_img])[:,0,:]
    print("Feature vectors genereated for all train vectors...")
    
    # Save train feature vectors
    h5f = h5py.File(features_path, 'w')
    h5f.create_dataset('dataset_1', data = features_unique_image_train)
    ls_image_names = [str.encode(name) for name in ls_image_names]
    h5f.create_dataset('dataset_2', data = ls_image_names)
    h5f.close()
#=============================================================
# Generate triplet candidates    
def generate_candidate_triplets(input_csv_path,output_csv_path,triplets_fixed_class,column_target):
    # Create output csv file for saving the triplets, at output_csv_path
    myFields = ["query_image","relevant_image","irrelevant_image","loss"]
    with open(output_csv_path, 'w') as csvFile:
        writer = csv.DictWriter(csvFile, delimiter=',', lineterminator='\n', fieldnames=myFields) 
        writer.writeheader()  
    
    #Import train data frame
    df = pd.read_csv(input_csv_path)

    # Extract list of images
    high_level_params=[column_target]
    df_std_image_1=df.groupby(high_level_params).size().reset_index(name='counts').sort_values(by=['counts'], ascending=False)
    ls_std_image_1=df_std_image_1[column_target].tolist()

    triplets_per_class=triplets_fixed_class
    # Extract triplets
    for item in ls_std_image_1:
        df_item = df[df[column_target]==item]
        ls_not_item=[e for e in ls_std_image_1 if not e in item]
        for not_item in ls_not_item: 
            df_not_item = df[df[column_target]==not_item]
            # Generate triplets_per_class triplets per image class pairs considered 
            for _ in range(triplets_per_class):
                query_name,relevant_name = df_item.sample(2)["image_name"]
                irrelevant_name = df_not_item.sample(1)["image_name"].iloc[0]
                #Save the triplets in the output csv file
                with open(output_csv_path, 'a') as csvFile:
                    writer = csv.DictWriter(csvFile, delimiter=',', lineterminator='\n', fieldnames=myFields)   
                    # Create dictionary for saving triplet
                    dict_triplet = {myFields[0]:query_name, myFields[1]:relevant_name, myFields[2]:irrelevant_name,myFields[3]:-1}
                    writer.writerow(dict_triplet)     
    print("Candidate triplets generated...")
#=============================================================    
# Compute triplet loss
def triplet_loss_compute(margin,user_latent,positive_item_latent,negative_item_latent):
    loss = np.maximum(0, margin + 
        np.sum(np.square(user_latent - positive_item_latent), axis=-1, keepdims=True) -
        np.sum(np.square(user_latent - negative_item_latent), axis=-1, keepdims=True))
    return loss
#=============================================================
# Generate csv of ranked triplets from csv of triplet candidates 
def generate_ranked_triplets(csv_path_triplets,csv_path_ranked_triplets,features_path,margin,verbose):
    # Read data frame of candidate triplets
    df_triplets = pd.read_csv(csv_path_triplets)

    # Read pre-generated features
    features_h5 = h5py.File(features_path, 'r')
    unique_features = np.array(features_h5.get('dataset_1'))
    unique_filenames = list(features_h5.get('dataset_2'))
    ls_filenames_unique=[filename.decode("utf-8") for filename in unique_filenames]
    features_h5.close()

    # Compute loss of each triplet
    for i in range(len(df_triplets)):
        if verbose == True:
            if ((i+1)%500) ==0:
                print("%s triplets evaluated..." %str(i+1))

        #Get image names from each image triplet
        q_name,r_name,irr_name = df_triplets.iloc[i]["query_image"], df_triplets.iloc[i]["relevant_image"], df_triplets.iloc[i]["irrelevant_image"]

        # Extract features of the images in the triplet
        try:
            q_idx=ls_filenames_unique.index(q_name)
            r_idx=ls_filenames_unique.index(r_name)
            irr_idx=ls_filenames_unique.index(irr_name)
            q_feature = unique_features[q_idx]
            r_feature = unique_features[r_idx]
            irr_feature = unique_features[irr_idx]

            #Compute and save loss of the triplet in consideration
            loss = triplet_loss_compute(margin,q_feature,r_feature,irr_feature)
            df_triplets.loc[i,"loss"]=loss
        except:
            if verbose == True:
                print("Candidate ignored...")
                
    print("Number of candidates with non-zero loss: ",sum(df_triplets['loss']!=0) )
    print("Total number of candidates: ",len(df_triplets) )
    #Rank the triplets
    df_ranked_triplets = df_triplets.sort_values(by=['loss'], axis=0, ascending=False, inplace=False)
    # Save the ranked triplets, to csv_path_ranked_triplets
    df_ranked_triplets.to_csv(csv_path_ranked_triplets)
    print("Candidate triplets ranked and saved, successfully...")
#=============================================================

