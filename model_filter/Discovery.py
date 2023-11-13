import numpy as np
from collections import Counter
import pandas as pd

def weeks_before(summary):
    week_identification_np=np.array([['B.1',11],['B.1.1 ',10],['B.1.177',44],['B.1.2',47],['B.1.1.7',56],['AY.44',82],['AY.43',105], ['AY.4',79],['AY.103',84],['B.1.617.2',87],['BA.1',107],['BA.2',111],['BA.2.9',111],['BA.2.3',121],['BA.2.12.1',126],['BA.5.1',134],['CH.1.1',156],['XBB.1.5',159]])
    # Convert the input summary to a NumPy array for easier processing.
    summary_np = np.array(summary)

    # Extract lineages from the summary array.
    Lineages = summary_np[:, 0]

    # Create a dictionary with counts of each lineage.
    Lineages_dict = Counter(Lineages)

    # Initialize a list to store the final output.
    final_distance = []

    # Iterate through each unique lineage in the dictionary.
    for k in Lineages_dict.keys():
        if k == 'unknown':
            continue

        # Find indices in summary where the current lineage is present.
        i_k = np.where(summary_np == k)[0]

        # Find the index in the predefined array for the current lineage.
        i_w = np.where(week_identification_np == k)[0]

        # Extract recognized weeks for the current lineage.
        week_recognize = np.array(list(map(int, week_identification_np[i_w, 1])))

        # Extract predicted counts and anomaly weeks for the current lineage from the summary.
        predicted = np.array(list(map(int, summary_np[i_k, 2])))
        week_an = np.array(list(map(int, summary_np[i_k, 3])))

        # Determine the first week when an anomaly was predicted.
        Index_first_prediction = np.where(predicted > 0)[0]
        if len(Index_first_prediction) == 0:
            continue
        week_first_prediction = min(list(week_an[Index_first_prediction]))
        week_first_prediction_true = week_first_prediction + 1

        # Calculate the difference in weeks between recognized and first predicted anomaly week.
        week_before = np.array(week_recognize - week_first_prediction_true)

        # Append the result to the final_distance list.
        summary = [k, week_before]
        final_distance.append(summary)

    # Return the list of differences for each lineage.
    return final_distance