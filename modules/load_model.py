from Deep_Retrieval_Siamese_Architecture import addition
from Deep_Retrieval_Siamese_Architecture import deep_retrieval_siamese
from Deep_Retrieval_Siamese_Architecture import TripletLoss
from keras import optimizers

def load_deep_retrieval_siamese(model_parameters, weights_path=None):
    batch_size = model_parameters["batch"]
    margin = model_parameters["margin"]
    model_type = model_parameters["model_type"]
    ## Load Deep Image Retrieval model
    print('Loading Deep Image Retrieval (pre-trained weights) model...')
    model = deep_retrieval_siamese((224, 224, 3), model_type)
    print('Done!')

    ## Freeze all layers
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[3].layers:
        layer.trainable = False
    
    ## Unfreeze pca layer, if instructed
    if model_parameters["tune_pca"] is True:
        for layer in model.layers:
            layer.trainable = True
    ## Unfreeze pca layer, if instructed
    if model_parameters["tune_conv5"] is True:
        for layer in model.layers[3].layers[-5:]:
            layer.trainable = True
    
    # Instantiate loss object
    tripletloss = TripletLoss(margin,batch_size)
    
    ## Compile model
    rate = model_parameters["lr"] #1e-5
    model.compile(loss=tripletloss.triplet_loss_deep_retrieval, optimizer=optimizers.Adam(lr=rate))
    
    ## Load weights, if provided. Otherwise, return model with pre-trained weights
    if weights_path is not None:
        model.load_weights(weights_path)

    return model

if __name__== "__main__":
    model_parameters={"tune_conv5":True, "tune_pca":True,
                     "lr":1e-5,"batch":64}
    model = load_deep_retrieval_siamese(model_parameters)
    model.summary()
