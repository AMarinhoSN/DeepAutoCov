import tensorflow as tf
## INPUT
# autoencoder: model defined in the script "model_dl.py"
# train1 e train2: training set defined in the script "main_prediction_AE.py"
# nb_epoch: number of epoch defined when you run the script "main_prediction_AE"
# batch_size: batch_size defined when you run the script "main_prediction_AE"
## OUTPUT
# history: is the model trained (used in the script "main_prediction_AE") 

def autoencoder_training_GPU(autoencoder, train1, train2, nb_epoch, batch_size):
    # Assicurarsi che sia disponibile una GPU
    if tf.config.experimental.list_physical_devices('GPU'):
        # Configurare TensorFlow per utilizzare la GPU
        with tf.device('/GPU:0'):
            history = autoencoder.fit(train1, train2,
                                      epochs=nb_epoch,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      verbose=1
                                      ).history
    else:
        print("No GPU available. Using CPU instead.")
        history = autoencoder.fit(train1, train2,
                                  epochs=nb_epoch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  verbose=1
                                  ).history

    return history
