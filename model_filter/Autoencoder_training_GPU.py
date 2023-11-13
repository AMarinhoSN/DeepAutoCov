import tensorflow as tf

def autoencoder_training_GPU(autoencoder, train1, train2, nb_epoch, batch_size):
    """
    This function trains an autoencoder model, utilizing a GPU if available.
    Parameters:
    - autoencoder: The autoencoder model to be trained.
    - train1: The input data for training.
    - train2: The target data for training (can be the same as train1 for autoencoders).
    - nb_epoch: The number of epochs for training.
    - batch_size: The size of batches used in training.

    Returns:
    - history: A history object containing the training progress information.
    """

    # Check if a GPU is available in the TensorFlow environment.
    if tf.config.experimental.list_physical_devices('GPU'):
        # If a GPU is available, configure TensorFlow to use it.
        with tf.device('/GPU:0'):
            # Train the autoencoder on the GPU using the specified training data, epochs, and batch size.
            # 'shuffle=True' shuffles the training data, and 'verbose=1' prints out the training progress.
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

    # Return the history object capturing the training progress.
    return history