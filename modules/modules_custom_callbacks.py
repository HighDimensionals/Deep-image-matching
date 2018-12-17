import keras
import pandas as pd
import numpy as np
from keras.preprocessing import image
from IPython.display import clear_output
from modules_generating_triplets import generate_candidate_triplets
from modules_generating_triplets import generate_ranked_triplets
from modules_generating_triplets import generate_feature_vectors_resampling
import matplotlib.pyplot as plt
#========================================================================
def translate_metric(x):
    translations = {'acc': "Accuracy", 'loss': "Model loss"}
    if x in translations:
        return translations[x]
    else:
        return x
#========================================================================
class computeTriplets(keras.callbacks.Callback):
    def __init__(self,epochs_th,features_path,csv_path_train_siamese,batch_size,imagesDir,margin,csv_path_ranked_triplets,csv_path_triplets,triplets_fixed_class,column_target,len_features):
        self.epoch_triplets=epochs_th
        self.features_path=features_path
        self.input_csv_path=csv_path_train_siamese
        self.batch_size=batch_size
        self.imagesDir=imagesDir
        self.margin=margin
        self.csv_path_ranked_triplets = csv_path_ranked_triplets
        self.csv_path_triplets=csv_path_triplets
        self.triplets_fixed_class=triplets_fixed_class
        self.column_target=column_target
        self.len_features=len_features
        return 
#========================================================================    
    def on_epoch_end(self,epoch=0,logs={}):
        if (epoch+1)%self.epoch_triplets==0:
            ## Generate and save feature vectors
            generate_feature_vectors_resampling(self.model,self.input_csv_path,self.imagesDir,self.features_path,len_features=self.len_features, batch_size=self.batch_size)

            ## Generate triplet candidates
            generate_candidate_triplets(self.input_csv_path,self.csv_path_triplets,self.triplets_fixed_class,self.column_target)

            ## Generate ranked triplets
            # Print something every 500 triplets
            verbose=False
            # Generate ranked triplets
            print("Ranking the triplets...")
            generate_ranked_triplets(self.csv_path_triplets,self.csv_path_ranked_triplets,self.features_path,self.margin,verbose)
#========================================================================
class PlotLosses(keras.callbacks.Callback):
    def __init__(self, figsize=None):
        super(PlotLosses, self).__init__()
        self.figsize = figsize

    def on_train_begin(self, logs={}):

        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs.copy())

        clear_output(wait=True)
        plt.figure(figsize=self.figsize)
        
        for metric_id, metric in enumerate(self.base_metrics):
            plt.subplot(1, len(self.base_metrics), metric_id + 1)
            
            plt.plot(range(1, len(self.logs) + 1),
                     [log[metric] for log in self.logs],
                     label="training")
            if self.params['do_validation']:
                plt.plot(range(1, len(self.logs) + 1),
                         [log['val_' + metric] for log in self.logs],
                         label="validation")
            plt.title(translate_metric(metric))
            plt.xlabel('epoch')
            plt.legend(loc='center left')
        
        plt.tight_layout()
        plt.show();
#========================================================================
def save_figs(history, tempStr, save_path):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('base accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.legend('train', loc='upper left')
    plt.savefig(save_path + tempStr + '_acc.jpg')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.legend('train', loc='upper left')
    plt.savefig(save_path + tempStr + '_loss.jpg')
    plt.show()
#========================================================================    
# Define data generator
from get_regions import rmac_regions, get_size_vgg_feat_map
from keras.preprocessing.image import ImageDataGenerator
def generator(input_csv_path,top_triplets, batch_size, img_height,img_width,channels,imagesDir,batches_per_epoch,epochs_per_triplets,
              len_features,augmentation_flag=False,rotation_max=90):

    df=pd.read_csv(input_csv_path)
    df=df[:top_triplets]
    # Number of triplet images
    n_samples=len(df)
   
    # Initialize query, relevant, irrelevant images and ROI batch
    batch_query = np.zeros((batch_size, img_height, img_width, channels))
    batch_relevant = np.zeros((batch_size, img_height, img_width, channels))
    batch_irrelevant = np.zeros((batch_size, img_height, img_width, channels))
    
    target = np.zeros([batch_size,3,len_features])#because target doesnt matter...
    #Initialize index
    idx=0
    count_epochs=0
    
    while True:
        # Extract data for this batch
        df_batch = df[idx*batch_size:idx*batch_size+batch_size]
        ls_query_batch=df_batch["query_image"].tolist()
        ls_relevant_batch=df_batch["relevant_image"].tolist()
        ls_irrelevant_batch=df_batch["irrelevant_image"].tolist()
        # If augmentation flag is true (works for catalogue images only)
        if augmentation_flag is True:
            datagen = ImageDataGenerator(rotation_range=rotation_max,fill_mode='constant',cval=255)
            for i in range(batch_size):
                # Extract query image
                img = image.load_img(imagesDir+"/"+ls_query_batch[i], target_size=(img_height, img_width))
                img = image.img_to_array(img)
                img = np.squeeze(datagen.flow(np.expand_dims(img,axis=0), batch_size=1).next().astype(int),axis=0)
                batch_query[i] = img/255            
                # Extract relevant image
                img = image.load_img(imagesDir+"/"+ls_relevant_batch[i], target_size=(img_height, img_width))
                img = image.img_to_array(img)
                img = np.squeeze(datagen.flow(np.expand_dims(img,axis=0), batch_size=1).next().astype(int),axis=0)
                batch_relevant[i] = img/255           
                # Extract irrelevant image
                img = image.load_img(imagesDir+"/"+ls_irrelevant_batch[i], target_size=(img_height, img_width))
                img = image.img_to_array(img)
                img = np.squeeze(datagen.flow(np.expand_dims(img,axis=0), batch_size=1).next().astype(int),axis=0)
                batch_irrelevant[i] = img/255       
        # If no augmentation required
        else:
            for i in range(batch_size):
                # Extract query image
                img = image.load_img(imagesDir+"/"+ls_query_batch[i], target_size=(img_height, img_width))
                img = image.img_to_array(img)
                batch_query[i] = img/255            
                # Extract relevant image
                img = image.load_img(imagesDir+"/"+ls_relevant_batch[i], target_size=(img_height, img_width))
                img = image.img_to_array(img)
                batch_relevant[i] = img/255           
                # Extract irrelevant image
                img = image.load_img(imagesDir+"/"+ls_irrelevant_batch[i], target_size=(img_height, img_width))
                img = image.img_to_array(img)
                batch_irrelevant[i] = img/255       
        #Increment idx for next batch of data    
        idx += 1
        
        #If we are at end of the data batch, repeat from beginning
        if idx== batches_per_epoch:
            idx = 0
            #Reload triplets
            df=pd.read_csv(input_csv_path)
            df=df[:top_triplets]        
        yield [batch_query, batch_relevant, batch_irrelevant],target                                
#========================================================================        
