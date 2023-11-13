import numpy as np


def find_lineage_per_week(dataset, week, dictionary_lineage_weeks):
    """
    This function filters a dataset to retrieve rows corresponding to specific lineages for a given week.

    Parameters:
    - dataset: The dataset to be filtered, assumed to be a NumPy array.
    - week: The specific week for which lineages are to be filtered.
    - dictionary_lineage_weeks: A dictionary mapping weeks to lineages.

    Returns:
    - results: A NumPy array containing the filtered dataset rows.
    """

    # Assume that the lineage column is the last column in the dataset.
    lineage_column = dataset.shape[1] - 1

    # Extract the lineages for the specified week from the dictionary.
    weekly_lineages = dictionary_lineage_weeks[week]

    # Create an empty NumPy array to store the results.
    results = np.empty((0, dataset.shape[1]), dtype=dataset.dtype)

    # Iterate through the dataset and select only the rows with lineages that correspond to the specified week.
    for lineage in weekly_lineages:
        # Find the rows in the dataset where the lineage matches.
        selected_rows = dataset[np.where(dataset[:, lineage_column] == lineage)]

        # Stack the selected rows onto the results array.
        results = np.vstack((results, selected_rows))

    # Return the array containing the selected rows.
    return results


def find_indices_lineage_per_week(lineage_column, week, dictionary_lineage_weeks):
    """
    This function finds row indices in a dataset for specific lineages corresponding to a given week.

    Parameters:
    - lineage_column: The column in the dataset containing lineage information.
    - week: The specific week for which lineages are to be matched.
    - dictionary_lineage_weeks: A dictionary mapping weeks to lineages.

    Returns:
    - indices_rows_np: A NumPy array of integers containing the indices of rows matching the specified lineages for the week.
    """

    # Extract the lineages for the specified week from the dictionary.
    weekly_lineages = dictionary_lineage_weeks[week]

    # Create an empty list to store the indices of corresponding rows.
    indices_rows = []

    # Iterate through the lineage column and select only the indices of rows with lineages corresponding to the specified week.
    for i, lineage in enumerate(lineage_column):
        if lineage in weekly_lineages:
            indices_rows.append(i)

    # Convert the indices list to a NumPy array of integers.
    indices_rows_np = np.array(indices_rows, dtype=int)

    # Return the array of indices.
    return indices_rows_np