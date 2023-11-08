def lookup(y_test_i_predict , y_test_step_i, knowledge,mse):
    prediction = y_test_i_predict # [-1,1]
    lineages = y_test_step_i
    mse = mse
    lineages_known = knowledge
    for i, lineage in enumerate(lineages):
        if lineage in lineages_known and prediction[i] == -1:
            prediction[i] = 1
            mse[i] = 0

    return prediction,mse


def lookup_post(y_test_i_predict , y_test_step_i, knowledge):
    prediction = y_test_i_predict # [-1,1]
    lineages = y_test_step_i
    lineages_known = knowledge
    for i, lineage in enumerate(lineages):
        if lineage in lineages_known and prediction[i] == -1:
            prediction[i] = 1

    return prediction