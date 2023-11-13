import numpy as np
from collections import Counter


def Compute_nseq_week(a):
    """
    This function calculates the number of weeks required to predict 100 variants of the same type as anomalous.
    Parameters:
    - a: A list containing sublists in the format ['lineage', predictions, week], where 'lineage' is the name of the lineage,
         'predictions' is the number of sequences defined as anomalous, and 'week' is the week of simulation.

    Returns:
    - A NumPy array containing sublists with the lineage name and the number of weeks to flag 100 sequences as anomalies.
    """

    # Convert the input list to a NumPy array for easier manipulation.
    a_np = np.array(a)
    Variants = a_np[:, 0]  # Extract lineage names.
    Prediction = a_np[:, 1]  # Extract prediction counts.
    Prediction_int = [int(x) for x in Prediction]  # Convert predictions to integers.
    Week = a_np[:, 2]  # Extract week numbers.

    # Count the occurrences of each variant.
    Variant = Counter(Variants)

    # Initialize a list to store the final summary.
    summary_Final = []

    # Iterate through each unique variant.
    for k in Variant.keys():
        # Find indices where the variant name matches 'k'.
        i_k = np.where(Variants == k)[0]
        Variant_counter = Prediction[i_k]  # Extract predictions for the current variant.
        Variant_counter_int = [int(x) for x in Variant_counter]  # Convert to integers.

        # Calculate the cumulative sum of predictions for the current variant.
        my_cum_sum_array = np.cumsum(Variant_counter_int)

        # Skip the variant if the total number of predictions is less than 100.
        if sum(Variant_counter_int) < 100:
            continue

        # Find the index where the cumulative sum reaches or exceeds 100.
        Index = np.where(my_cum_sum_array >= 100)
        Interest_index = np.array(Index)
        Interest_index_min = Interest_index[:, 0]
        Index_to_week = i_k[Interest_index_min]  # Map the index to the week number.
        week_objective = int(Week[Index_to_week])  # Get the objective week number.

        # Find the week number where the variant first appears.
        Index_start = np.array(np.where(Variants == k))
        week_start = int(Week[Index_start[:, 0]])

        # Calculate the number of weeks to reach 100 predictions and add to the summary.
        summary = [k, week_objective - week_start]
        summary_Final.append(summary)

    # Return the final summary as a NumPy array.
    return np.array(summary_Final)