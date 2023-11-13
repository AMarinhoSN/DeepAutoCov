# funxione che mi mappa i lineage
def map_lineage_to_finalclass(class_list, non_neutral):
    """
    This function maps a list of lineages to a final class based on their neutrality.
    Parameters:
    - class_list: A list of lineages to be classified.
    - non_neutral: A list of lineages that are considered non-neutral.

    Returns:
    - final_class_list: A list where each lineage from class_list is classified as -1 (non-neutral) or 1 (neutral).
    """

    # Initialize an empty list to store the final classification of each lineage.
    final_class_list = []

    # Iterate over each lineage in the class_list.
    for c in class_list:
        # Check if the current lineage is in the list of non-neutral lineages.
        if c in non_neutral:
            # If it is non-neutral, append -1 to the final_class_list.
            final_class_list.append(-1)
        else:
            # If it is not non-neutral (i.e., it is neutral), append 1 to the final_class_list.
            final_class_list.append(1)

    # Return the list containing the final classification for each lineage.
    return final_class_list