import numpy as np
from collections import Counter

def Calcolo_week(a):
    a_np = np.array(a)
    Varianti = a_np[:, 0]
    Prediction = a_np[:, 1]
    Prediction_int = [int(x) for x in Prediction]
    Week = a_np[:, 2]
    print(a_np[:, 1])
    Variant = Counter(Varianti)
    new_list = []
    Riassunto_Finale = []
    for k in Variant.keys():
        i_k = np.where(Varianti == k)[0]
        Variant_counter = Prediction[i_k]
        Variant_counter_int = [int(x) for x in Variant_counter]
        my_cum_sum_array = np.cumsum(Variant_counter_int)
        if sum(Variant_counter_int) < 100:
            continue

        Index = np.where(my_cum_sum_array >= 100)
        Interest_index = np.array(Index)
        Interest_index_min = Interest_index[:, 0]
        Index_to_week = i_k[Interest_index_min]
        Settimana_obiettivo = int(Week[Index_to_week])
        Indice_inizio = np.array(np.where(Varianti == k))
        Settimana_inizio = int(Week[Indice_inizio[:, 0]])
        riassunto = [k, Settimana_obiettivo - Settimana_inizio]
        Riassunto_Finale.append(riassunto)
    return np.array(Riassunto_Finale)
