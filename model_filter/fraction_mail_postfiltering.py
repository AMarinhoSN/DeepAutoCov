import numpy as np
import pandas as pd

# Top100: Function to calculate the number of True Positives and False Positives among the 100 sequences with the highest Mean Squared Error (MSE) values.
# INPUT
#     1)mse: list of Mean Squared Error (MSE)
#     2)y_test_step_i: type of lineages in test set
#     3)week: week of simulation
#     4)threshold: threshold
#     5)n: 100
# OUTPUT
#    1) FP: False Positive
#    2) TP: True Positive
#    3) N: number of sequences

def Top100(mse, y_test_step_i, week, soglia, n, lineage_known):
    settimana_giusta = week + 1
    mse_var = []
    FP_TOT = []
    TP_TOT = []
    for i in range(0, len(mse)):
        mse_var.append([mse[i], y_test_step_i[i]])
    mse_var.sort(key=lambda x: x[0], reverse=True)  # lo ordino
    mse_var_np = np.array(mse_var)
    mse_np = np.array(list(map(float, mse_var_np[:, 0])))
    if week <= 10:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1

    if week >10 and week <=11:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1.1':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1

    if week > 11 and week <= 44:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1

    if week > 44 and week <= 47:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[i] == 'B.1.177':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1

    if week > 47 and week <= 56:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[i] == 'B.1.2' or selection[i] == 'B.1.177':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1

    if week > 56 and week <= 79:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' :
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1

    if week > 79 and week <= 82:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[i] == 'AY.4':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1

    if week > 82 and week <= 84:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[i] == 'AY.4' or selection[i] == 'AY.44':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1

    if week > 84 and week <= 87:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[i] == 'AY.4' or selection[i] == 'AY.103' or selection[i] == 'AY.44':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1

    if week > 87 and week <= 105:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[i] == 'AY.4' or selection[i] == 'B.1.617.2'  or selection[i] == 'AY.44' or selection[i] == 'AY.103':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1
    if week > 105 and week <= 107:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[
                i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[
                i] == 'AY.4' or selection[i] == 'B.1.617.2' or selection[i] == 'AY.43' or selection[
                i] == 'AY.44' or selection[i] == 'AY.103':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1

    if week > 107 and week <= 111:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[
                i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[
                i] == 'AY.4' or selection[i] == 'B.1.617.2' or selection[i] == 'AY.43' or selection[
                i] == 'AY.44' or selection[i] == 'AY.103' or selection[i] == 'BA.1':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1
    if week > 111 and week <= 121:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[
                i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[
                i] == 'AY.4' or selection[i] == 'B.1.617.2' or selection[i] == 'AY.43' or selection[
                i] == 'AY.44' or selection[i] == 'AY.103' or selection[i] == 'BA.1' or selection[
                i] == 'BA.2' or selection[i] == 'BA.2.9':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1
    if week > 121 and week <= 126:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[
                i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[
                i] == 'AY.4' or selection[i] == 'B.1.617.2' or selection[i] == 'AY.43' or selection[
                i] == 'AY.44' or selection[i] == 'AY.103' or selection[i] == 'BA.1' or selection[
                i] == 'BA.2' or selection[i] == 'BA.2.9' or selection[i] == 'BA.2.3':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1
    if week > 126 and week <= 134:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[
                i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[
                i] == 'AY.4' or selection[i] == 'B.1.617.2' or selection[i] == 'AY.43' or selection[
                i] == 'AY.44' or selection[i] == 'AY.103' or selection[i] == 'BA.1' or selection[
                i] == 'BA.2' or selection[i] == 'BA.2.9' or selection[i] == 'BA.2.3' or selection[
                i] == 'BA.2.12.1':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1
    if week > 134 and week <= 156:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[
                i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[
                i] == 'AY.4' or selection[i] == 'B.1.617.2' or selection[i] == 'AY.43' or selection[
                i] == 'AY.44' or selection[i] == 'AY.103' or selection[i] == 'BA.1' or selection[
                i] == 'BA.2' or selection[i] == 'BA.2.9' or selection[i] == 'BA.2.3' or selection[
                i] == 'BA.2.12.1' or selection[i] == 'BA.5.1' :
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1
    if week > 156 and week <= 159:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[
                i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[
                i] == 'AY.4' or selection[i] == 'B.1.617.2' or selection[i] == 'AY.43' or selection[
                i] == 'AY.44' or selection[i] == 'AY.103' or selection[i] == 'BA.1' or selection[
                i] == 'BA.2' or selection[i] == 'BA.2.9' or selection[i] == 'BA.2.3' or selection[
                i] == 'BA.2.12.1' or selection[i] == 'BA.5.1' or selection[i] == 'CH.1.1':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1
    if week > 159 and week <= 160:
        i_anomalie_mse = np.where(mse_np > soglia)  # trovo gli indici delle anomalie vere
        mse_var_np_filtr = mse_var_np[i_anomalie_mse, :]  # adesso ho il mse ordinato e soprasoglia
        if n > len(mse_var_np_filtr[0, :, 0]):
            n = len(mse_var_np_filtr[0, :, 0])
        selection = mse_var_np_filtr[0, 0:n, :]
        filtered_indices = [i for i, item in enumerate(selection[:, 1]) if item not in lineage_known]
        selection = [selection[i, 1] for i in filtered_indices]
        if n > len(selection):
            n = len(selection)
        FP = 0
        TP = 0
        for i in range(0, n):
            if not selection:  # Se la lista è vuota
                break
            if selection[i] == 'unknown' or selection[i] == 'B.1' or selection[i] == 'B.1.1' or selection[
                i] == 'B.1.2' or selection[i] == 'B.1.177' or selection[i] == 'B.1.1.7' or selection[
                i] == 'AY.4' or selection[i] == 'B.1.617.2' or selection[i] == 'AY.43' or selection[
                i] == 'AY.44' or selection[i] == 'AY.103' or selection[i] == 'BA.1' or selection[
                i] == 'BA.2' or selection[i] == 'BA.2.9' or selection[i] == 'BA.2.3' or selection[
                i] == 'BA.2.12.1' or selection[i] == 'BA.5.1' or selection[i] == 'CH.1.1' or selection[i] == 'XB.1.5':
                FP = FP + 1  # SE HO MESSO COME
            else:
                TP = TP + 1
    return FP, TP, n,
