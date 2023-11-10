import numpy as np

def compute_confusion_matrix(true_class, mse, threshold):
    """
    Compute confusion matrix based on given threshold
    """
    if mse > threshold and true_class == -1:
        return (1, 0, 0, 0)  # TP
    elif mse <= threshold and true_class == 1:
        return (0, 1, 0, 0)  # TN
    elif mse > threshold and true_class == 1:
        return (0, 0, 1, 0)  # FP
    else:
        return (0, 0, 0, 1)  # FN

def evaluate_thresholds(true_classes, mses, init_threshold, min_threshold, max_threshold):
    """
    Evaluate different thresholds to obtain the 
    """
    step = 0.03 * init_threshold
    TP_TOT, TN_TOT, FP_TOT, FN_TOT = 0, 0, 0, 0

    for threshold in np.arange(min_threshold, max_threshold, step):
        for true_class, mse in zip(true_classes, mses):
            TP, TN, FP, FN = compute_confusion_matrix(true_class, mse, threshold)
            TP_TOT += TP
            TN_TOT += TN
            FP_TOT += FP
            FN_TOT += FN

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
    Soglia_info=[]
    for threshold in np.linspace(min_threshold-0.01, max_threshold+0.01, 40):
        TP_TOT, TN_TOT, FP_TOT, FN_TOT = 0, 0, 0, 0
        for true_class, mse in zip(true_classes, mses):
            TP, TN, FP, FN = compute_confusion_matrix(true_class, mse, threshold)
            TP_TOT += TP
            TN_TOT += TN
            FP_TOT += FP
            FN_TOT += FN
        l=[threshold,TP_TOT/(TP_TOT+FP_TOT+1),TP_TOT/(TP_TOT+FN_TOT+1),TP_TOT,TN_TOT,FP_TOT,FN_TOT]
        Soglia_info.append(l)
    return Soglia_info
