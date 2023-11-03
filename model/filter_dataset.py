import numpy as np

def trova_lineage_per_settimana(dataset, settimana, dizionario_lineage_settimane):
    # Assume that the lineage column is the last one.
    colonna_lineage = dataset.shape[1] - 1

    # We extract the lineages for the specified week from the dictionary.
    lineage_settimanali = dizionario_lineage_settimane[settimana]

    # We create an empty ndarray to store the results.
    risultati = np.empty((0, dataset.shape[1]), dtype=dataset.dtype)

    # We iterate through the dataset and select only the rows with lineages corresponding to the specified week
    for lineage in lineage_settimanali:
        righe_selezionate = dataset[np.where(dataset[:, colonna_lineage] == lineage)]
        risultati = np.vstack((risultati, righe_selezionate))

    return risultati


def trova_indici_lineage_per_settimana(column_lineage, week, dictionary_lineage_week):
    # trova_indici_lineage_per_settimana : This function find the index of elements for new training set for each retraining week
    # INPUT: 
    #    1) column_lineage: lineages lineages examined up to the training week
    #    2) week: week of retraining
    # OUTPUT:
    #    1) index_raw_np: indexes for new training set
    # We extract the lineages for the specified week from the dictionary.
    lineage_week = dictionary_lineage_week[week]

    # We create an empty list to store the indexes of the corresponding rows.
    index_raw = []

    # We iterate through the lineage column and select only the row indices with lineages corresponding to the specified week
    for i, lineage in enumerate(column_lineage):
        if lineage in lineage_week:
            index_raw.append(i)

    # Converts indexes to an ndarray of integers.
    index_raw_np = np.array(index_raw , dtype=int)

    return index_raw_np


