import numpy as np
from datetime import datetime
import random

def filter_row_by_column_length(ndarray, string_list, col_index, target_len):
    mask = np.array([len(s) == target_len for s in ndarray[:,col_index]], dtype=bool)
    return ndarray[mask], [s for s, m in zip(string_list, mask) if m], ndarray[np.logical_not(mask)], [s for s, m in zip(string_list, mask) if not m]


# filter_row_by_column_length_sostitution : find the sequences correctly with corrects date 
def filter_row_by_column_length_sostitution(ndarray, string_list, col_index, target_len):

    def check_and_add_suffix(s):
        suffixes = ["-01", "-07", "-10", "-14", "-20", "-25", "-27"]
        if len(s) == 7:
            extended_s = s + random.choice(suffixes)
            if len(extended_s) == target_len:
                return extended_s
        return s

    extended_strings = np.array([check_and_add_suffix(s) for s in ndarray[:, col_index]])
    mask = np.array([len(s) == target_len for s in extended_strings], dtype=bool)
    ndarray[:, col_index] = extended_strings

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

def select_rows(ndarray,country):
    selected_rows = []
    for row in ndarray:
        if isinstance(row[6], str) and country in row[6]:
            selected_rows.append(row)
    return np.array(selected_rows)

# to create dataset
def select_rows_dataset(ndarray, country):
    selected_rows = []
    selected_indices = []

    for index, row in enumerate(ndarray):
        if isinstance(row[6], str) and country in row[6]:
            selected_rows.append(row)
            selected_indices.append(index)

    return np.array(selected_rows), selected_indices
