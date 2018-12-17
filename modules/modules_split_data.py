from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import csv
####################################################################
# Module for checking if file exists -- NOT USING, CURRENTLY
def existed_file(row):
    my_file = './Real_resized_Images/'+ row['image_name']
    if Path(my_file).is_file():
        try:
            im = Image.open(my_file)
            return True
        except:
            return False
    else:
        return False
####################################################################
# Module for splitting csv into train, validation and test
def split_train_val_test(input_csv_path,csv_path_train,csv_path_val,csv_path_test,column_target,samples_per_class_val,samples_per_class_test,min_count_th,imagesDir,ls_ignore_class=[]):

    # Read data frame
    df=pd.read_csv(input_csv_path)

    # Create output csv files 
    myFields = list(df)
    with open(csv_path_train, 'w') as csvFile:
        writer = csv.DictWriter(csvFile, delimiter=',', lineterminator='\n', fieldnames=myFields) 
        writer.writeheader()  
    with open(csv_path_val, 'w') as csvFile:
        writer = csv.DictWriter(csvFile, delimiter=',', lineterminator='\n', fieldnames=myFields) 
        writer.writeheader()          
    with open(csv_path_test, 'w') as csvFile:
        writer = csv.DictWriter(csvFile, delimiter=',', lineterminator='\n', fieldnames=myFields) 
        writer.writeheader()  

    # Rows corresponding to non-existent images are removed from df
    ls_remove_idx=[]
    for row in df.iterrows():
        my_file = imagesDir + '/'+ row[1]['image_name']
        if Path(my_file).is_file():
            try:
                im = Image.open(my_file)
            except:
                ls_remove_idx.append(row[0])
        else:
            ls_remove_idx.append(row[0])
    df = df.drop(ls_remove_idx)
    
    # Extract list of images
    min_count=min_count_th #minimum number of samples required, in dataset, for that image to be considered
    high_level_params=[column_target]
    df_std_image_1=df.groupby(high_level_params).size().reset_index(name='counts').sort_values(by=['counts'], ascending=False)
    ls_std_image_1=df_std_image_1[df_std_image_1["counts"]>min_count][column_target].tolist()
    
    # Remove any list items not to be included
    for item in ls_ignore_class:
        ls_std_image_1.remove(item)
    
    # Create data frame for train, validation and test
    df_train = pd.DataFrame(columns=myFields)
    df_validation = pd.DataFrame(columns=myFields)    
    df_test=pd.DataFrame(columns=myFields)

    #For each class, split data into train and test
    for item in ls_std_image_1:
        df_item = df[df[column_target]==item]
        df_item_val_test = df_item.sample(samples_per_class_val+samples_per_class_test)
        df_test = df_test.append(df_item_val_test[:samples_per_class_test])
        df_validation = df_validation.append(df_item_val_test[samples_per_class_test:])
        df_train = df_train.append(df_item.drop(df_item_val_test.index))

    # Save the train, validation and test data frames
    df_train.to_csv(csv_path_train)
    df_validation.to_csv(csv_path_val)
    df_test.to_csv(csv_path_test)
    
    #Print data sizes
    print("Classes: ", ls_std_image_1)
    print("Number of classes: ", len(ls_std_image_1))
    print("Number of training samples: ",len(df_train))
    print("Number of validation samples: ",len(df_validation))
    print("Number of test samples: ",len(df_test))
####################################################################
