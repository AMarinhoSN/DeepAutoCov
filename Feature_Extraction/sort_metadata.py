import numpy as np
from datetime import datetime
import random

def filter_row_by_column_length(ndarray, string_list, col_index, target_len):
    # This function filters rows in a NumPy array and a parallel string list based on a length criterion.
    # Parameters:
    # ndarray: A NumPy array where each row is an entry.
    # string_list: A list of strings, parallel to the ndarray rows.
    # col_index: The index of the column in ndarray to be checked for length.
    # target_len: The target length for the strings in the specified column.

    # Create a boolean mask where each entry is True if the length of the string
    # in the specified column equals the target length, and False otherwise.
    mask = np.array([len(s) == target_len for s in ndarray[:, col_index]], dtype=bool)

    # Apply the mask to the ndarray to filter rows where the condition is True.
    # Also, create a filtered list from string_list where corresponding ndarray rows meet the condition.
    filtered_ndarray = ndarray[mask]
    filtered_string_list = [s for s, m in zip(string_list, mask) if m]

    # Apply the inverse of the mask to ndarray to get rows where the condition is False.
    # Similarly, create a list from string_list where corresponding ndarray rows do not meet the condition.
    inverse_filtered_ndarray = ndarray[np.logical_not(mask)]
    inverse_filtered_string_list = [s for s, m in zip(string_list, mask) if not m]

    # Return the filtered ndarray and string list (where condition is True) and
    # the inverse-filtered ndarray and string list (where condition is False).
    return filtered_ndarray, filtered_string_list, inverse_filtered_ndarray, inverse_filtered_string_list

# filter_row_by_column_length_sostitution : find the sequences correctly with corrects date 
def filter_row_by_column_length_sostitution(ndarray, string_list, col_index, target_len):
    # This function is designed to filter rows of a NumPy array (ndarray) and a parallel list (string_list)
    # based on the length of the strings in a specified column of the ndarray after potentially modifying them.
    # Parameters:
    # ndarray: The NumPy array to be processed.
    # string_list: A list of strings parallel to the ndarray rows.
    # col_index: The index of the column in ndarray to check and potentially modify string length.
    # target_len: The target length of strings after modification.

    def check_and_add_suffix(s):
        # A nested function to check the length of a string and add a suffix if necessary.
        # suffixes: A list of predefined suffixes to be randomly added to the string.
        suffixes = ["-01", "-07", "-10", "-14", "-20", "-25", "-27"]
        # If the string length is 7, a random suffix is added.
        if len(s) == 7:
            extended_s = s + random.choice(suffixes)
            # The modified string is returned if it matches the target length.
            if len(extended_s) == target_len:
                return extended_s
        # If no modification is done, the original string is returned.
        return s

    # Apply check_and_add_suffix to each string in the specified column of ndarray.
    extended_strings = np.array([check_and_add_suffix(s) for s in ndarray[:, col_index]])

    # Create a boolean mask where True represents rows where the modified string length matches target_len.
    mask = np.array([len(s) == target_len for s in extended_strings], dtype=bool)

    # Update the specified column in ndarray with the modified strings.
    ndarray[:, col_index] = extended_strings

    # Return filtered ndarray and string_list based on the mask, and their inverses.
    # The ndarray and string_list are filtered to include only rows/strings where the modified string length matches the target.
    # The inverses include rows/strings where the modified string length does not match the target.
    return ndarray[mask], [s for s, m in zip(string_list, mask) if m], ndarray[np.logical_not(mask)], [s for s, m in zip(string_list, mask) if not m]

def insert_sequence_as_column(data, dates, sequence):
    """
    It inserts the amino acid sequence as a column of an ndarray that contains the metadata,
    then sorting the ndarray by dates (in ascending order).

    Args:
        data (ndarray): the ndarray containing the metadata
        dates (list): the list of dates corresponding to each row in the ndarray.
        sequence (list): the list of the amino acid sequence to be entered as a column.

    Returns:
        ndarray: the ndarray sorted by dates, with the amino acid sequence inserted as a column.
    """
    # Transform date in object datetime
    date_objs = np.array([datetime.strptime(date, '%Y-%m-%d') for date in dates])

    # Add the sequence column as the last column of the ndarray
    data_with_sequence = np.column_stack((data, sequence))

    # Add a column with datetime objects of the dates
    data_with_dates = np.column_stack((data_with_sequence, date_objs))

    # Sort the ndarray by dates
    sorted_indices = np.argsort(data_with_dates[:, -1])
    sorted_data = data_with_dates[sorted_indices]

    # Delete the date column
    sorted_data = np.delete(sorted_data, -1, axis=1)

    return sorted_data

def select_rows(ndarray, country):
    # This function is designed to filter rows from a NumPy array based on a country criterion.
    # It takes two parameters:
    # ndarray: A NumPy array representing a dataset where each row is an entry.
    # country: A string representing the country used as the filtering criterion.

    # Initialize an empty list to store the selected rows.
    selected_rows = []

    # Iterate over each row in the ndarray.
    for row in ndarray:
        # Check if the 7th element (index 6) of the row is a string and contains the specified country.
        if isinstance(row[6], str) and country in row[6]:
            # If the row meets the criteria, append it to the list of selected rows.
            selected_rows.append(row)

    # Convert the list of selected rows back into a NumPy array and return it.
    return np.array(selected_rows)

# Function to create a subset of a dataset based on a specified country.
def select_rows_dataset(ndarray, country):
    # Initialize empty lists to store the selected rows and their indices.
    selected_rows = []
    selected_indices = []

    # Iterate over each row in the dataset along with its index.
    for index, row in enumerate(ndarray):
        # Check if the 7th element of the row (index 6) is a string and contains the specified country.
        if isinstance(row[6], str) and country in row[6]:
            # If the condition is met, append the entire row to 'selected_rows'.
            selected_rows.append(row)
            # Append the current index to 'selected_indices'.
            selected_indices.append(index)
    # Return the filtered rows as a NumPy array and the list of their indices.
    return np.array(selected_rows), selected_indices
