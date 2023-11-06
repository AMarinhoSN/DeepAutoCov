import tensorflow as tf
# model: function to create the moel
# INPUT:
#    1) input_dim: dimension of input 
#    2) encodin_dim: dimension of encoding
#    3) hidden_dim: dimension of deep layers
#    4) reduction_factor
#    5) path_salvataggio_file: path where to save the model
# OUTPUT:
#    1) the model
def model(input_dim,encoding_dim,hidden_dim_1,hidden_dim_2,hidden_dim_3,hidden_dim_4,hidden_dim_5,reduction_factor,path_salvataggio_file):
    # Input Layer
    input_layer = tf.keras.layers.Input(shape=(input_dim,))

    # Encoder
    encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh",
                                    activity_regularizer=tf.keras.regularizers.l2(reduction_factor))(input_layer)
    encoder = tf.keras.layers.Dropout(0.3)(encoder)  # reduce overfitting
    encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_3, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(hidden_dim_4, activation='relu')(encoder)

    # Strato centrale con rumore
    latent = tf.keras.layers.Dense(hidden_dim_5, activation=tf.nn.leaky_relu)(encoder)
    noise_factor = 0.1
    latent_with_noise = tf.keras.layers.Lambda(
        lambda x: x + noise_factor * tf.keras.backend.random_normal(shape=tf.shape(x)))(latent)

    # Decoder
    decoder = tf.keras.layers.Dense(hidden_dim_4, activation='relu')(latent_with_noise)
    decoder = tf.keras.layers.Dense(hidden_dim_3, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(hidden_dim_2, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(decoder)
    decoder = tf.keras.layers.Dropout(0.3)(decoder)
    decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)

    # Output
    decoder = tf.keras.layers.Dense(input_dim, activation='tanh',
                                    activity_regularizer=tf.keras.regularizers.l2(reduction_factor))(decoder)

    # Autoencoder
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
    autoencoder.summary()

    # Define the callbacks for checkpoints and early stopping
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=path_salvataggio_file + "/autoencoder_AERNS.h5",
                                            mode='min', monitor='loss', verbose=2, save_best_only=True)
    # define our early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=0.001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True)
    # compile
    autoencoder.compile(metrics=['mse'],
                        loss='mean_squared_error',
                        optimizer='adam')

    return autoencoder
