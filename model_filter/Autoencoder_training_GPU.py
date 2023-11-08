import tensorflow as tf

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