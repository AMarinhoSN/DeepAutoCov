import tensorflow as tf

def model(input_dim, encoding_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4, hidden_dim_5, reduction_factor, path_salvataggio_file):
    """
    This function creates and returns an autoencoder model.
    Parameters:
    - input_dim: Dimension of the input layer.
    - encoding_dim: Dimension of the encoding layer.
    - hidden_dim_1 to hidden_dim_5: Dimensions of the hidden layers.
    - reduction_factor: Factor used for L2 regularization to reduce overfitting.
    - path_salvataggio_file: Path where the model checkpoint will be saved.

    Returns:
    - autoencoder: The constructed TensorFlow Keras autoencoder model.
    """

    # Input Layer
    # Create the input layer with the specified dimension.
    input_layer = tf.keras.layers.Input(shape=(input_dim,))

    # Encoder
    # Construct the encoder part of the autoencoder using dense layers, dropout for regularization,
    # and various activation functions.
    encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(reduction_factor))(input_layer)
    encoder = tf.keras.layers.Dropout(0.3)(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_3, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_4, activation='relu')(encoder)

    # Central Layer with Noise
    # Add a latent layer with leaky ReLU activation and introduce noise to make the model more robust.
    latent = tf.keras.layers.Dense(hidden_dim_5, activation=tf.nn.leaky_relu)(encoder)
    noise_factor = 0.1
    latent_with_noise = tf.keras.layers.Lambda(
        lambda x: x + noise_factor * tf.keras.backend.random_normal(shape=tf.shape(x)))(latent)

    # Decoder
    # Build the decoder part of the autoencoder to reconstruct the input from the encoded representation.
    decoder = tf.keras.layers.Dense(hidden_dim_4, activation='relu')(latent_with_noise)
    decoder = tf.keras.layers.Dense(hidden_dim_3, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(decoder)
    decoder = tf.keras.layers.Dropout(0.3)(decoder)
    decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)

    # Output Layer
    # Define the output layer to reconstruct the original input.
    decoder = tf.keras.layers.Dense(input_dim, activation='tanh', activity_regularizer=tf.keras.regularizers.l2(reduction_factor))(decoder)

    # Autoencoder Model
    # Define the autoencoder model that maps the input to its reconstruction.
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()

    # Callbacks for Model Checkpoint and Early Stopping
    # Set up a checkpoint to save the model and early stopping to prevent overfitting.
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=path_salvataggio_file + "/autoencoder_fraud_AERNS.h5",
                                            mode='min', monitor='loss', verbose=2, save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)

    # Compile the Autoencoder
    # Use mean squared error as the loss function and Adam optimizer for training.
    autoencoder.compile(metrics=['mse'],
                        loss='mean_squared_error',
                        optimizer='adam')

    # Return the compiled autoencoder model.
    return autoencoder
