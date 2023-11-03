import numpy as np
from collections import Counter
import pandas as pd
# scoperta : calculate how soon we discover the lineages
#INPUT:
#    1) measure_sensibilit: list that contains [['name_of_lineage',total_sequence, prediced_anomaly,week]] where total_sequence is the number of sequence in the week and prediced_anomaly is the number of sequences predicted as anomaly
#OUTPUT
#    1) final_distance: list that for each lineages contain the number of weeks before that the model identified as anomaly 

def scoperta(measure_sensibilit):
    final_distance=[]
    week_identified_np=np.array([['B.1',11],['B.1.1 ',10],['B.1.177',44],['B.1.2',47],['B.1.1.7',56],['AY.44',82],['AY.43',105], ['AY.4',79],['AY.103',84],['B.1.617.2',87],['BA.1',107],['BA.2',111],['BA.2.9',111],['BA.2.3',121],['BA.2.12.1',126],['BA.5.1',134],['CH.1.1',156],['XBB.1.5',159]])
    measure_sensibilit_np = np.array(measure_sensibilit) 
    Varianti = measure_sensibilit_np[:, 0] #select the lineages 
    variant_dict = Counter(Varianti) 
    for k in variant_dict.keys(): 
        if k == 'unknown':
            continue
        i_k = np.where(measure_sensibilit_np == k)[0] 
        i_w = np.where(week_identified_np == k)[0]
        week_identified= np.array(list(map(int, week_identified_np[i_w, 1]))) 
        predetti = np.array(list(map(int, measure_sensibilit_np[i_k, 2]))) #prediction
        week_an = np.array(list(map(int, measure_sensibilit_np[i_k, 3]))) #week
        Index_first_detection=np.where(predetti>0)[0] 
        if len(Index_first_detection)==0:
            continue
        week_fist_detection=min(list(week_an[Index_first_detection]))
        week_fist_detection_true=week_fist_detection+1
        distance=np.array(week_identified-week_fist_detection_true)
        summary=[k,distance]
        final_distance.append(summary)
    return final_distance
