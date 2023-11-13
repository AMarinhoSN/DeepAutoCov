import numpy as np

def compute_confusion_matrix(true_class, mse, threshold):
    """
    Compute confusion matrix based on given threshold
    """
    # This function computes the elements of a confusion matrix (True Positive, True Negative,
    # False Positive, False Negative) for a single prediction against a specified threshold.
    # Parameters:
    # true_class: The actual class of the data point (-1 or 1).
    # mse: The Mean Squared Error of the prediction.
    # threshold: The threshold value to determine the classification.

    # Check if the MSE is greater than the threshold and the true class is -1 (non-neutral or negative class).
    if mse > threshold and true_class == -1:
        return (1, 0, 0, 0)  # This is a True Positive (TP) scenario.

    # Check if the MSE is less than or equal to the threshold and the true class is 1 (neutral or positive class).
    elif mse <= threshold and true_class == 1:
        return (0, 1, 0, 0)  # This is a True Negative (TN) scenario.

    # Check if the MSE is greater than the threshold and the true class is 1 (neutral or positive class).
    elif mse > threshold and true_class == 1:
        return (0, 0, 1, 0)  # This is a False Positive (FP) scenario.

    # If none of the above conditions are met, it is a False Negative (FN) scenario.
    else:
        return (0, 0, 0, 1)  # FN

# The function returns a tuple representing the count of TP, TN, FP, FN for the given data point.

def evaluate_thresholds(true_classes, mses, init_threshold, min_threshold, max_threshold):
    """
    Evaluate different thresholds to obtain the 
    """
    # This function evaluates various threshold values to determine the performance of a classification model.
    # Parameters:
    # true_classes: A list of the true classes for each data point (e.g., 1 for positive, -1 for negative).
    # mses: A list of mean squared errors (MSE) for each prediction.
    # init_threshold: An initial threshold value, used to set the step size for iterating over thresholds.
    # min_threshold: The minimum threshold value to start evaluation from.
    # max_threshold: The maximum threshold value for evaluation.

    # Set the step size for threshold iteration as 3% of the initial threshold.
    step = 0.03 * init_threshold

    # Initialize counters for True Positives, True Negatives, False Positives, and False Negatives.
    TP_TOT, TN_TOT, FP_TOT, FN_TOT = 0, 0, 0, 0

    # Iterate over a range of threshold values.
    for threshold in np.arange(min_threshold, max_threshold, step):
        # For each threshold, evaluate the confusion matrix for each data point.
        for true_class, mse in zip(true_classes, mses):
            # Compute the confusion matrix for each data point and threshold.
            TP, TN, FP, FN = compute_confusion_matrix(true_class, mse, threshold)
            # Accumulate the counts of TP, TN, FP, FN.
            TP_TOT += TP
            TN_TOT += TN
            FP_TOT += FP
            FN_TOT += FN

    # Return the total counts of TP, TN, FP, FN over all thresholds.
    return TP_TOT, TN_TOT, FP_TOT, FN_TOT

def evaluate_pcr(true_classes, mses, init_threshold, min_threshold, max_threshold):
    """
    Evaluate different thresholds to obtain the precision and recall. 
    # INPUT
    #    1)true_classes: list that contains true class
    #    2)mses: list that contains mse
    #    3)init_threshold: treshold for autoencoder
    #    4)min_threshold: minimum of mse
    #    5)max_threshold: maximum of mse
    # OUTPUT
    #    Soglia_info: list that for each treshold contains the precision and recall
    """
    step = 0.03 * init_threshold
    passi=40
    PRC_info=[]
    for threshold in np.linspace(min_threshold-0.01, max_threshold+0.01, 40):
        TP_TOT, TN_TOT, FP_TOT, FN_TOT = 0, 0, 0, 0
        for true_class, mse in zip(true_classes, mses):
            TP, TN, FP, FN = compute_confusion_matrix(true_class, mse, threshold)
            TP_TOT += TP
            TN_TOT += TN
            FP_TOT += FP
            FN_TOT += FN
        l=[threshold,TP_TOT/(TP_TOT+FP_TOT+1),TP_TOT/(TP_TOT+FN_TOT+1),TP_TOT,TN_TOT,FP_TOT,FN_TOT]
        PRC_info.append(l)
    return PRC_info
