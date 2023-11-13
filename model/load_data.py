import pandas as pd
import os

def load_data(dir_dataset, week_range):
    # This function is designed to load data from a specified directory based on a range of weeks.
    # Parameters:
    # dir_dataset: The directory containing the dataset.
    # week_range: A list of week numbers (integers) specifying the range of weeks to be loaded.

    # Convert week numbers to strings for comparison with directory names.
    week_range = [str(x) for x in week_range]

    # List the folders in dir_dataset that match the specified week range.
    weeks_folder = [x for x in os.listdir(dir_dataset) if x in week_range]

    # Initialize empty lists to store dataframes and week identifiers.
    df_list = []
    w_list = []

    # Iterate through each week folder.
    for week in weeks_folder:
        # Construct the path to the dataset file for the week.
        df_path = dir_dataset + week + '/week_dataset.txt'

        # Load the dataset from the file into a pandas DataFrame.
        df = pd.read_csv(df_path, header=None)

        # Append the DataFrame to the df_list.
        df_list.append(df)

        # Extend w_list with the week identifier, repeated for the number of rows in df.
        w_list += [week] * df.shape[0]

        # The following block appears to be unused and could be a remnant from previous code iterations.
        # It sets a directory path and walks through its structure, but does not perform any operations on the files.
        directory = os.path.join("c:\\", "path")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    f = open(file, 'r')
                    f.close()

    # Concatenate all DataFrames in df_list and return it along with the week list (w_list).
    return pd.concat(df_list), w_list
