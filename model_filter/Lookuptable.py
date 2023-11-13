def lookup(y_test_i_predict, y_test_step_i, knowledge, mse):
    """
    This function modifies predictions and Mean Squared Error (MSE) values based on prior knowledge of lineages.

    Parameters:
    - y_test_i_predict: A list/array of initial predictions, where -1 and 1 represent different classes.
    - y_test_step_i: A list/array of lineages corresponding to each prediction.
    - knowledge: A list of lineages that are known or considered significant.
    - mse: A list/array of Mean Squared Error (MSE) values corresponding to each prediction.

    Returns:
    - prediction: The modified list of predictions.
    - mse: The modified list of Mean Squared Error (MSE) values.
    """

    # Assign the initial predictions to a variable.
    prediction = y_test_i_predict

    # Assign the lineages to a variable.
    lineages = y_test_step_i

    # Assign the MSE values to a variable.
    mse = mse

    # Assign the known lineages to a variable.
    lineages_known = knowledge

    # Iterate through each lineage in the list of lineages.
    for i, lineage in enumerate(lineages):
        # Check if the lineage is in the list of known lineages and if the corresponding prediction is -1.
        if lineage in lineages_known and prediction[i] == -1:
            # If both conditions are met, modify the prediction for this lineage to 1.
            prediction[i] = 1
            # Also, set the corresponding MSE value to 0.
            mse[i] = 0

    # Return the modified list of predictions and MSE values.
    return prediction, mse


def lookup_post(y_test_i_predict, y_test_step_i, knowledge):
    """
    This function modifies predictions based on prior knowledge of lineages.

    Parameters:
    - y_test_i_predict: A list/array of initial predictions, where -1 and 1 represent different classes.
    - y_test_step_i: A list/array of lineages corresponding to each prediction.
    - knowledge: A list of lineages that are known or considered important.

    Returns:
    - prediction: The modified list of predictions.
    """

    # Assign the initial predictions to a variable.
    prediction = y_test_i_predict

    # Assign the lineages to a variable.
    lineages = y_test_step_i

    # Assign the known lineages to a variable.
    lineages_known = knowledge

    # Iterate through each lineage in the list of lineages.
    for i, lineage in enumerate(lineages):
        # Check if the lineage is in the list of known lineages and if the corresponding prediction is -1.
        if lineage in lineages_known and prediction[i] == -1:
            # If both conditions are met, modify the prediction for this lineage to 1.
            prediction[i] = 1

    # Return the modified list of predictions.
    return prediction