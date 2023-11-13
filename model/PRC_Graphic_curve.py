import matplotlib.pyplot as plt

def compute_prc(lists, path_save):
    """
    This function calculates and plots the Precision-Recall Curve (PRC).
    Parameters:
    - lists: A list containing sublists with precision and recall values for each threshold.
    - path_save: The path where the PRC image will be saved.

    Returns:
    - somme_precision: A list of average precision values at different thresholds.
    - somme_recall: A list of average recall values at different thresholds.
    - info_graph: A string indicating the completion of the graphing process.
    """

    # Initialize two lists to accumulate precision and recall sums for 40 columns.
    somme_precision = [0] * 40
    somme_recall = [0] * 40

    # Iterate through each sublist in the provided list.
    for sottolista in lists:
        # For each sublist, iterate through its 40 elements.
        for i in range(40):
            # Extract the sub-sublist for each column.
            sotto_sottolista = sottolista[i]

            # Add the precision and recall values to their respective sums.
            somme_precision[i] += sotto_sottolista[1]
            somme_recall[i] += sotto_sottolista[2]

    # Normalize the sums by dividing by the number of elements (assumed to be 16 here).
    for i in range(40):
        somme_precision[i] = somme_precision[i] / 16
        somme_recall[i] = somme_recall[i] / 16

    # Plotting the Precision-Recall Curve.
    plt.figure(1)
    plt.plot(somme_recall, somme_precision, '-', label='Autoencoder')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(str(path_save) + '/PRC.jpg', bbox_inches='tight')
    plt.show()

    # Indicate that the graphing process is complete.
    info_graph = 'done'

    # Return the average precision and recall values, and the graph information.
    return somme_precision, somme_recall, info_graph


