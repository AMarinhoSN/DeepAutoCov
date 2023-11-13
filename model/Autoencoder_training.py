## INPUT
# autoencoder: model defined in the script "model_dl.py"
# train1 e train2: training set defined in the script "main_prediction_AE.py"
# nb_epoch: number of epoch defined in the script "main_prediction_AE"
# batch_size: batch_size defined in the script "main_prediction_AE"
## OUTPUT
# history: model trained (used in the script "main_prediction_AE") 

def Autoencoder_training(autoencoder, train1,train2,nb_epoch,batch_size):
    history = autoencoder.fit(train1, train2,
                                  epochs=nb_epoch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  #validation_data=(test_data, test_data),
                                  verbose=1
                                  ).history
    return history
