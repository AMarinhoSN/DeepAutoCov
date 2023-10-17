from scipy.stats import shapiro
import numpy as np

def test_normality(autoencoder, train_model):
    # Calcola le previsioni dell'autoencoder sui dati di addestramento
    predictions = autoencoder.predict(train_model)

    # Calcola l'MSE dell'autoencoder sui dati di addestramento
    mse = np.mean(np.power(train_model - predictions, 2), axis=1)


    # Testa la normalità dell'MSE utilizzando il test di Shapiro-Wilk
    if len(mse)>3:
        _, p_value = shapiro(mse.flatten())
    else:
        p_value=-1

    return p_value,mse
