def Autoencoder_training(autoencoder, train1,train2,nb_epoch,batch_size):
    history = autoencoder.fit(train1, train2,
                                  epochs=nb_epoch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  #validation_data=(test_data, test_data),
                                  verbose=1
                                  ).history
    return history
