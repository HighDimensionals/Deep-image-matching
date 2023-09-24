# Deep Image Matching 

This is a Keras implementation of a deep image matching scheme, where the image descriptors(features) are computed using the concatenation of a VGG16 network (without the fully connected layers), a pooling layer, normalization layer, PCA layer, and another normalization layer. During the training phase, the VGG16 weights and PCA layer weights are fine-tuned using a Siamese neural network: the Siamese neural network is composed of three identical copies of the deep network described above. The training (triplet) loss defined with the Siamese network allows for similar images to be closer and dissimilar images to be further apart in the descriptor/feature space. This training architecture uses ideas from the following papers:

https://arxiv.org/abs/1511.05879
https://arxiv.org/abs/1510.07493
https://arxiv.org/abs/1604.01325

Training
==================
The fine-tuning of the Siamese neural network weights can be performed by running the demo "FineTuning_main.ipynb".
For generating training data, importing the network architecture, and fine-tuning the network weights, the following modules are called by the "FineTuning_main.ipynb" demo file, sequentially. 
- "modules/modules_split_data" splits input data into train, test and validation data
- "modules/load_model.py" loads the Siamese network by calling "modules/Deep_Retrieval_Siamese_Architecture.py" 
- "modules/modules_generating_triplets.py" generates triplets of training images, ranked in order of decreasing hardness. Each triplet consists of an "anchor image", "relevant image" that is similar to the anchor image, and an "irrelevant image" that is dissimilar to the anchor image. The closer the "irrelevant image" to the "anchor image", or the further the "relevant image" to the "anchor image", the more the harness of a given triplet of images, and vice versa.
- "modules/modules_custom_callbacks.py" contains the custom callback modules that are called at the end of every epoch of training. This modules within are called to resample the training triplets, reset the training data generator, etc, after every epoch.

All the modules called within the demo "FineTuning_main.ipynb" are placed in the "/modules" directory.

Testing
==================
Run "Test_gen_features_main.ipynb" to obtain the test results. The demo file (i) loads the descriptor weights, (ii) computes and saves the features for all database images, (iii) finds the closest matching image in the database for each test query image, and (iv) reports the results in terms of "mean average precision" (mAP). All the modules called within "Test_gen_features_main.ipynb" are in the  

In this architecture, essentially, three-identical copies of the NN architecture presented in https://arxiv.org/abs/1511.05879 are created. Specifically, in each of the three identical networks, (i) the image is passed as input to VGG16 network, (ii) the activations from the last layer of VGG16 network are either max-pooled or sum-pooled, (iii) the pooled activations are passed through a PCA layer, (iv) the activations of the PCA layer are used as image descriptors after further normalization. The three identical networks share the VGG16 weights as well as the PCA layer weights.
Triplets of images are used for training the triplet network, such that there is a (i) base image, (ii) similar image, (iii) dissimilar image in each triplet of images. Correspondingly, a notion of triplet loss is formulated on top of these three identical networks such that the network weights are updated for (i) similar images to have smaller loss and (ii) dissimilar images to have larger loss. 

All the modules called within the demo "Test_gen_features_main.ipynb" are placed in the "/modules/modules_generating_results.py" file.
