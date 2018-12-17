from keras.preprocessing import image
import numpy as np
import pandas as pd
import os

###############################################
#Modules for generating results
###############################################
# Computing and saving feature vectors
## INPUTS: 
# imagesDir: a directory of images 
# input_csv_path: csv of image names in the imagesDir
# column_target: the column in the input_csv_path that is being considered for classification/ranking
# features_parent_dir: directory for saving all the features
# features_save_path: path for saving the computed features
## OUTPUT:
# features_array: array of features
def compute_save_features(input_csv_path,imagesDir,model,features_parent_dir,features_save_path,column_target,len_features,verbose=False):
    # Read data frame
    df_database=pd.read_csv(input_csv_path)
    # Compute feature vectors
    ls_filenames_database = df_database["image_name"].tolist()
    ls_targets = df_database[column_target].tolist()
    len_database = len(ls_filenames_database)
    features_array = np.zeros((len_database, len_features))
    for i in range(len_database):
        if i%500 ==0 and verbose==True:
            print(str(i) + " features computed and saved...")
        img = image.load_img(imagesDir+"/"+ls_filenames_database[i], target_size=(224, 224))
        img = image.img_to_array(img)
        img = img/255
        features_array[i]=model.predict([np.expand_dims(img,axis=0),np.expand_dims(img,axis=0),np.expand_dims(img,axis=0)])[0,0,:]
    np.save(features_parent_dir+'/'+features_save_path, features_array)
    return features_array
###############################################
def compute_avg_precision(y_preds,y_label):
    if y_label in y_preds:
        return 1.0
    else:
        return 0
###############################################
def compute_avg_precision_original(y_preds,y_label):
    sum_precision=0
    j=0
    for i in range(len(y_preds)):
        if y_preds[i]==y_label:
            j+=1
            sum_precision+=float(j)/(i+1)
        else:
            sum_precision+=float(j)/(i+1)
    score = float(sum_precision)/len(y_preds)
    return score
###############################################
def save_similar_images(targetDir,ls_similar_image_names,original_image_name,imagesDir):
    #create targetDir directory if it does not exist
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    #Save the similar images
    i=0
    for image_name in ls_similar_image_names:
        img = cv2.imread(imagesDir+"/"+image_name)
        i+=1
        cv2.imwrite(targetDir+"/"+str(i)+"_"+image_name,img)
    #Save original image as well
    img = cv2.imread(imagesDir+"/"+original_image_name)
    i+=1
    cv2.imwrite(targetDir+"/"+image_name,img)
###############################################
def compute_save_results(no_top_similar_images,col_names,features_database,features_test,df_database,df_test,column_target,
                        results_parent_dir,results_dir,results_csv_path,save_images=False):
    df_results = pd.DataFrame(columns=col_names)
    ls_filenames_test = df_test["image_name"].tolist()
    ls_shapes_test = df_test[column_target].tolist()
    for i in range(len(ls_filenames_test)):
        similarities=np.dot(features_database,features_test[i].T)
        idx_top_sim=np.argsort(-similarities)[:no_top_similar_images]
        ls_top_shapes = df_database[column_target].iloc[idx_top_sim].tolist()
        ls_top_image_names = df_database["image_name"].iloc[idx_top_sim].tolist()
        score_ap = compute_avg_precision(ls_top_shapes,ls_shapes_test[i])#compute average precision
        #Append data frame
        dict_item={col_names[0]:ls_filenames_test[i],
        		  col_names[1]:ls_shapes_test[i],
                  col_names[2]:ls_top_shapes,
                  col_names[3]:score_ap}
        item_series = pd.Series(data=dict_item)
        df_results = df_results.append(item_series,ignore_index=True)
        #Save the most similar images
        targetDir=results_parent_dir+ results_dir +'/'+str(ls_filenames_test[i])
        imagesDir='./unique'
        if save_images is True:
            save_similar_images(targetDir,ls_top_image_names,ls_filenames_test[i],imagesDir)
    #Save the results data frame
    df_results.to_csv(results_parent_dir+ results_dir+results_csv_path)
###############################################
    