import numpy as np
from collections import Counter
import pandas as pd

def discovery(measure_sensitivity):
    """
    This function calculates how soon lineages are discovered as anomalies by the model.
    
    Inputs:
    - measure_sensitivity: A list containing [['name_of_lineage', total_sequence, predicted_anomaly, week]], where total_sequence is the number of sequences in the week, and predicted_anomaly is the number of sequences predicted as anomalies in that week.

    Outputs:
    - final_distance: A list containing, for each lineage, the number of weeks before the model identified it as an anomaly.
    """

    # Initialize a list to store the final distance for each lineage.
    final_distance = []

    # Predefined array containing lineages and the week they were identified.
    week_identified_np = np.array([...])

    # Convert measure_sensitivity to a NumPy array for easier processing.
    measure_sensitivity_np = np.array(measure_sensitivity)

    # Extract lineages from the measure_sensitivity array.
    Variants = measure_sensitivity_np[:, 0]

    # Create a dictionary with counts for each variant.
    variant_dict = Counter(Variants)

    # Iterate through each unique variant in the dictionary.
    for k in variant_dict.keys():
        if k == 'unknown':
            continue

        # Find indices in measure_sensitivity where the current lineage is present.
        i_k = np.where(measure_sensitivity_np == k)[0]

        # Find the index in the predefined array for the current lineage.
        i_w = np.where(week_identified_np == k)[0]

        # Extract the weeks when the lineage was identified.
        week_identified = np.array(list(map(int, week_identified_np[i_w, 1])))

        # Extract the number of predicted anomalies and their respective weeks.
        predicted = np.array(list(map(int, measure_sensitivity_np[i_k, 2])))
        week_an = np.array(list(map(int, measure_sensitivity_np[i_k, 3])))

        # Determine the first week when an anomaly was detected.
        Index_first_detection = np.where(predicted > 0)[0]
        if len(Index_first_detection) == 0:
            continue
        week_first_detection = min(list(week_an[Index_first_detection]))
        week_first_detection_true = week_first_detection + 1

        # Calculate the difference in weeks between the identified week and the first detection week.
        distance = np.array(week_identified - week_first_detection_true)

        # Append the result to the final_distance list.
        summary = [k, distance]
        final_distance.append(summary)

    # Return the list of distances for each lineage.
    return final_distance
