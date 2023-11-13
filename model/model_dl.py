import tensorflow as tf

def model(input_dim, encoding_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4, hidden_dim_5, reduction_factor, path_salvataggio_file):
    """
    This function creates and returns an autoencoder model.
    Parameters:
    - input_dim: Dimension of the input layer.
    - encoding_dim: Dimension of the encoding layer.
    - hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4, hidden_dim_5: Dimensions of the hidden layers.
    - reduction_factor: Regularization factor to reduce overfitting.
    - path_salvataggio_file: Path to save the trained model.

    Returns:
    - autoencoder: The constructed autoencoder model.
    """

    # Input Layer
    # Create the input layer with the specified input dimension.
    input_layer = tf.keras.layers.Input(shape=(input_dim,))

    # Encoder
    # Build the encoder part of the autoencoder with dense layers and dropout to reduce overfitting.
    encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh",
                                    activity_regularizer=tf.keras.regularizers.l2(reduction_factor))(input_layer)
    encoder = tf.keras.layers.Dropout(0.3)(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_3, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_4, activation='relu')(encoder)

    # Central Layer with Noise
    # Add a layer with leaky ReLU activation and inject noise to make the model robust.
    latent = tf.keras.layers.Dense(hidden_dim_5, activation=tf.nn.leaky_relu)(encoder)
    noise_factor = 0.1
    latent_with_noise = tf.keras.layers.Lambda(
        lambda x: x + noise_factor * tf.keras.backend.random_normal(shape=tf.shape(x)))(latent)

    # Decoder
    # Build the decoder part of the autoencoder to reconstruct the input.
    decoder = tf.keras.layers.Dense(hidden_dim_4, activation='relu')(latent_with_noise)
    decoder = tf.keras.layers.Dense(hidden_dim_3, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(decoder)
    decoder = tf.keras.layers.Dropout(0.3)(decoder)
    decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)

    # Output Layer
    # Define the output layer to match the input dimension.
    decoder = tf.keras.layers.Dense(input_dim, activation='tanh',
                                    activity_regularizer=tf.keras.regularizers.l2(reduction_factor))(decoder)

    # Construct the Autoencoder Model
    # The model takes the input and provides the decoded output.
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()

    # Set up callbacks for saving the model and early stopping.
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=path_salvataggio_file + "/autoencoder_AERNS.h5",
                                            mode='min', monitor='loss', verbose=2, save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)

    # Compile the Autoencoder
    # Use mean squared error as the loss function and Adam optimizer.
    autoencoder.compile(metrics=['mse'],
                        loss='mean_squared_error',
                        optimizer='adam')

    # Return the compiled autoencoder model.
    return autoencoder
