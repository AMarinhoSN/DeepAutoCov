from scipy.stats import shapiro
import numpy as np

def test_normality(autoencoder, train_model):
    # Calculates autoencoder predictions on training data
    predictions = autoencoder.predict(train_model)

    # Calculates the MSE of the autoencoder on the training data.
    mse = np.mean(np.power(train_model - predictions, 2), axis=1)


    # Test the normality of the MSE using the Shapiro-Wilk test.
    if len(mse)>3:
        _, p_value = shapiro(mse.flatten())
    else:
        p_value=-1

    return p_value,mse
