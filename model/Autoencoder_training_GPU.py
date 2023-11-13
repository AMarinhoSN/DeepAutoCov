import tensorflow as tf

def autoencoder_training_GPU(autoencoder, train1, train2, nb_epoch, batch_size):
    # This function trains an autoencoder model, preferably using a GPU if available.
    # Parameters:
    # autoencoder: The autoencoder model to be trained.
    # train1: The input data for training.
    # train2: The output data for training (can be the same as input for autoencoders).
    # nb_epoch: The number of epochs for training.
    # batch_size: The size of the batch used in training.

    # Check if a GPU is available for TensorFlow.
    if tf.config.experimental.list_physical_devices('GPU'):
        # If a GPU is available, configure TensorFlow to use the GPU.
        with tf.device('/GPU:0'):
            # Train the autoencoder on the GPU with the specified parameters.
            history = autoencoder.fit(train1, train2,
                                      epochs=nb_epoch,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      verbose=1
                                      ).history
    else:
        # If no GPU is available, print a message and use the CPU for training.
        print("No GPU available. Using CPU instead.")
        history = autoencoder.fit(train1, train2,
                                  epochs=nb_epoch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  verbose=1
                                  ).history

    # Return the training history object, which contains information about the training process.
    return history
