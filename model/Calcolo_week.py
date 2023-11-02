import numpy as np
from collections import Counter

# Calcolo_week : function that tells me how many weeks I need to predict 100 variants of the same type as anomalous
# INPUT
#    1)list that contains [['lineage1',predictions,week1],['lineage2',predictions,week1],['lineage2',predictions,week2]]. Where 'lineage' is the name of lineage, prediction is the number of predicte, prediction is the number of sequences defined as anomalous, week of simulation
# OUTPUT
#    2) list that contains the name of lineage and number of weeks to flag 100 sequences like anomalies 

def Calcolo_week(a):
    a_np = np.array(a)
    Variants = a_np[:, 0]
    Prediction = a_np[:, 1]
    Prediction_int = [int(x) for x in Prediction]
    Week = a_np[:, 2]
    print(a_np[:, 1])
    Variant = Counter(Variants)
    new_list = []
    summary_Final = []
    for k in Variant.keys():
        i_k = np.where(Variants == k)[0]
        Variant_counter = Prediction[i_k]
        Variant_counter_int = [int(x) for x in Variant_counter]
        my_cum_sum_array = np.cumsum(Variant_counter_int)
        if sum(Variant_counter_int) < 100:
            continue

        Index = np.where(my_cum_sum_array >= 100)
        Interest_index = np.array(Index)
        Interest_index_min = Interest_index[:, 0]
        Index_to_week = i_k[Interest_index_min]
        week_objective = int(Week[Index_to_week])
        Index_start = np.array(np.where(Variants == k))
        week_start = int(Week[Index_start[:, 0]])
        summary = [k, week_objective - week_start]
        summary_Final.append(summary)
    return np.array(summary_Final)
