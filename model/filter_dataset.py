import numpy as np

def trova_lineage_per_settimana(dataset, settimana, dizionario_lineage_settimane):
    # Assumiamo che la colonna dei lineage sia l'ultima
    colonna_lineage = dataset.shape[1] - 1

    # Estraiamo i lineage per la settimana specificata dal dizionario
    lineage_settimanali = dizionario_lineage_settimane[settimana]

    # Creiamo un ndarray vuoto per memorizzare i risultati
    risultati = np.empty((0, dataset.shape[1]), dtype=dataset.dtype)

    # Iteriamo attraverso il dataset e selezioniamo solo le righe con i lineage corrispondenti alla settimana specificata
    for lineage in lineage_settimanali:
        righe_selezionate = dataset[np.where(dataset[:, colonna_lineage] == lineage)]
        risultati = np.vstack((risultati, righe_selezionate))

    return risultati


def trova_indici_lineage_per_settimana(colonna_lineage, settimana, dizionario_lineage_settimane):
    # Estraiamo i lineage per la settimana specificata dal dizionario
    lineage_settimanali = dizionario_lineage_settimane[settimana]

    # Creiamo una lista vuota per memorizzare gli indici delle righe corrispondenti
    indici_righe = []

    # Iteriamo attraverso la colonna dei lineage e selezioniamo solo gli indici delle righe con i lineage corrispondenti alla settimana specificata
    for i, lineage in enumerate(colonna_lineage):
        if lineage in lineage_settimanali:
            indici_righe.append(i)

    # Converte gli indici in un ndarray di interi
    indici_righe_np = np.array(indici_righe, dtype=int)

    return indici_righe_np

# # Dataset di esempio (senza la colonna delle settimane)
# # Le colonne rappresentano: ID, Valore, Lineage
# dataset = np.array([
#     [1, 10, 'A'],
#     [2, 20, 'B'],
#     [3, 15, 'A'],
#     [4, 25, 'C'],
#     [5, 30, 'B'],
#     [6, 40, 'A'],
#     [7, 35, 'C'],
# ], dtype=object)
#
# # Dizionario di esempio: mappa le settimane ai gruppi di lineage
# dizionario_lineage_settimane = {
#     1: ['A', 'B'],
#     2: ['C'],
# }
#
# # Chiamiamo la funzione per trovare i lineage nella settimana 1
# risultati_settimana_1 = trova_lineage_per_settimana(dataset, 1, dizionario_lineage_settimane)
#
# # Visualizziamo il risultato
# print("Risultati per la settimana 1:")
# print(risultati_settimana_1)
#
# # Chiamiamo la funzione per trovare i lineage nella settimana 2
# risultati_settimana_2 = trova_lineage_per_settimana(dataset, 2, dizionario_lineage_settimane)
#
# # Visualizziamo il risultato
# print("\nRisultati per la settimana 2:")
# print(risultati_settimana_2)
