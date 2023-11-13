import numpy as np

def find_index_lineages_for_week(column_lineage, week, dictionary_lineage_week):
    """
    This function finds the indices of elements for a new training set for each retraining week.
    Parameters:
    - column_lineage: A list or array of lineage information examined up to the training week.
    - week: The specific week of retraining.
    - dictionary_lineage_week: A dictionary mapping weeks to specific lineages.

    Returns:
    - index_raw_np: An array of indices for the new training set.
    """

    # Extract the specific lineages for the given week from the dictionary.
    lineage_week = dictionary_lineage_week[week]

    # Create an empty list to store indices of rows whose lineages correspond to the specified week.
    index_raw = []

    # Iterate through each lineage in the column_lineage.
    for i, lineage in enumerate(column_lineage):
        # If the lineage is one of those designated for the specified week, add its index to the list.
        if lineage in lineage_week:
            index_raw.append(i)

    # Convert the list of indices to a NumPy array of integers.
    index_raw_np = np.array(index_raw, dtype=int)

    # Return the array of indices.
    return index_raw_np


