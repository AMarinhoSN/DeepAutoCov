import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_confusion_matrix import *
#from measure_of_variants import *

def falsepositive(measure_sensibilit, retraining_week, path_salvataggio):
    
    measure_sensibilit_np = np.array(measure_sensibilit)  
    Varianti = measure_sensibilit_np[:, 0]
    Casi = np.array(list(map(int, measure_sensibilit_np[:, 1])))
    Predetti = np.array(list(map(int, measure_sensibilit_np[:, 2])))
    Settimane = np.array(list(map(int, measure_sensibilit_np[:, 3])))
    Settimane_giuste = Settimane + 1  # metto le settimane giuste
    variant_dict = Counter(Varianti)
    i = 0
    FP_RATE_FINAL = []
    Final = []
    TP_FINAL = []  # CONTENITORE TP
    TN_FINAL = []  # CONTENITORE TN
    FP_FINAL = []  # CONTENITORE FP
    FN_FINAL = []  # CONTENITORE FN

    for k in retraining_week:
        i_k = np.where(((Settimane_giuste >= i) & (Settimane_giuste < k)))
        Variant_in_week_range = Varianti[i_k]  
        Casi_in_week_range = Casi[i_k] 
        Predetti_in_week_range = Predetti[i_k]  
        Settimane_in_week_range = Settimane_giuste[i_k]  
        # SETTIMANA DALLA 2 ALLA 10
        if k == 10:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l] 
                    i_anomalie = np.where(
                        Variant_in_week != 'unknown')  
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(Variant_in_week == 'unknown')
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_27 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_27)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list.")
                break  

        # SETTIMANA DALLA 10 ALLA 11
        if k == 11:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i) 
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((Variant_in_week == 'B.1') | (
                                               Variant_in_week == 'B.1.177') | (Variant_in_week == 'B.1.1.7') | (
                                               Variant_in_week == 'B.1.2') | (Variant_in_week == 'AY.44') | (
                                                   Variant_in_week == 'AY.43') | (Variant_in_week == 'AY.4') | (
                                                   Variant_in_week == 'AY.103') | (Variant_in_week == 'B.1.617.2') | (
                                                   Variant_in_week == 'BA.1') | (
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.9') | (Variant_in_week == 'BA.2') | (Variant_in_week == 'BA.2.12.1') | (
                                                   Variant_in_week == 'BA.2') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((Variant_in_week == 'unknown') | (Variant_in_week == 'B.1.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_35 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_35)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list.")
                break  

        # DALLA 11 ALLA 44
        if k == 44:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l] 
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((
                            Variant_in_week == 'B.1.177') | (Variant_in_week == 'B.1.1.7') | (
                                                   Variant_in_week == 'B.1.2') | (Variant_in_week == 'AY.44') | (
                                                   Variant_in_week == 'AY.43') | (Variant_in_week == 'AY.4') | (
                                                   Variant_in_week == 'AY.103') | (Variant_in_week == 'B.1.617.2') | (
                                                   Variant_in_week == 'BA.1') | (
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.9') | (Variant_in_week == 'BA.2') | (
                                                       Variant_in_week == 'BA.2.12.1') | (
                                                   Variant_in_week == 'BA.2') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((Variant_in_week == 'B.1.1') | (Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_45 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_45)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break  
        # DALLA 44 ALLA 47
        if k == 47:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((
                                                   Variant_in_week == 'B.1.2') | (Variant_in_week == 'B.1.1.7') | (Variant_in_week == 'AY.44') | (
                                                   Variant_in_week == 'AY.43') | (Variant_in_week == 'AY.4') | (
                                                   Variant_in_week == 'AY.103') | (Variant_in_week == 'B.1.617.2') | (
                                                   Variant_in_week == 'BA.1') | (
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.9') | (Variant_in_week == 'BA.2') | (
                                                   Variant_in_week == 'BA.2.12.1') | (
                                                   Variant_in_week == 'BA.2') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((
                                                   Variant_in_week == 'B.1.177') | (Variant_in_week == 'B.1.1') | (Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_48 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_48)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # DALLA 47 ALLA 56
        if k == 56:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((Variant_in_week == 'B.1.1.7') | (
                                                       Variant_in_week == 'AY.44') | (
                                                   Variant_in_week == 'AY.43') | (Variant_in_week == 'AY.4') | (
                                                   Variant_in_week == 'AY.103') | (Variant_in_week == 'B.1.617.2') | (
                                                   Variant_in_week == 'BA.1') | (
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.9') | (Variant_in_week == 'BA.2') | (
                                                   Variant_in_week == 'BA.2.12.1') | (
                                                   Variant_in_week == 'BA.2') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((
                                                   Variant_in_week == 'B.1.177') |(
                                                 Variant_in_week == 'B.1.2') | (Variant_in_week == 'B.1.1') | (
                                                     Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_49 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_49)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # DALLA 56 ALLA 79
        if k == 79:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l] 
                    i_anomalie = np.where(((
                                                   Variant_in_week == 'AY.44') | (
                                                   Variant_in_week == 'AY.43') | (Variant_in_week == 'AY.4') | (
                                                   Variant_in_week == 'AY.103') | (Variant_in_week == 'B.1.617.2') | (
                                                   Variant_in_week == 'BA.1') | (
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.9') | (Variant_in_week == 'BA.2') | (
                                                   Variant_in_week == 'BA.2.12.1') | (
                                                   Variant_in_week == 'BA.2') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.177') | (
                                                 Variant_in_week == 'B.1.2') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_51 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_51)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # DALLA 79 ALLA 82
        if k == 82:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l] 
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((
                                                   Variant_in_week == 'AY.44') | (
                                                   Variant_in_week == 'AY.43') | (
                                                   Variant_in_week == 'AY.103') | (Variant_in_week == 'B.1.617.2') | (
                                                   Variant_in_week == 'BA.1') | (
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.9') | (Variant_in_week == 'BA.2') | (
                                                   Variant_in_week == 'BA.2.12.1') | (
                                                   Variant_in_week == 'BA.2') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((Variant_in_week == 'AY.4') | (Variant_in_week == 'B.1.1.7') | (
                            Variant_in_week == 'B.1.177') | (
                                                 Variant_in_week == 'B.1.2') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_62 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_62)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # Dalla 82 alla 84
        if k == 84:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((Variant_in_week == 'B.1.617.2') |  (
                                                   Variant_in_week == 'AY.43') | (
                                                   Variant_in_week == 'AY.103') | (
                                                   Variant_in_week == 'BA.1') | (
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.9') | (Variant_in_week == 'BA.2') | (
                                                   Variant_in_week == 'BA.2.12.1') | (
                                                   Variant_in_week == 'BA.2') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((
                                                   Variant_in_week == 'AY.44') | (Variant_in_week == 'AY.4') | (Variant_in_week == 'B.1.1.7') | (
                            Variant_in_week == 'B.1.177') | (
                                                 Variant_in_week == 'B.1.2') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_75 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_75)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list.")
                break
        # Settimana da 84 a 87
        if k == 87:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((Variant_in_week == 'B.1.617.2') | (
                                                   Variant_in_week == 'AY.43') | (
                                                   Variant_in_week == 'BA.1') | (
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.9') | (Variant_in_week == 'BA.2') | (
                                                   Variant_in_week == 'BA.2.12.1') | (
                                                   Variant_in_week == 'BA.2') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((
                                                   Variant_in_week == 'AY.44') | (
                                                   Variant_in_week == 'AY.103') | (Variant_in_week == 'AY.4') | (
                                Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.177') | (
                                                 Variant_in_week == 'B.1.2') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_76 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_76)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break

        #Dalla 87 alla 105
        if k == 105:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l] 
                    i_anomalie = np.where(((
                                                 Variant_in_week == 'AY.43') | (
                                                   Variant_in_week == 'BA.1') | (
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.9') | (Variant_in_week == 'BA.2') | (
                                                   Variant_in_week == 'BA.2.12.1') | (
                                                   Variant_in_week == 'BA.2') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((
                                                   Variant_in_week == 'AY.103') | (
                                                 Variant_in_week == 'AY.44') | (Variant_in_week == 'B.1.617.2') | (
                                                     Variant_in_week == 'AY.4') | (
                                                 Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.177') | (
                                                 Variant_in_week == 'B.1.2') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_77 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_77)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        #dalla 105 alla 107
        if k == 107:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((Variant_in_week == 'BA.2') | (
                                                 Variant_in_week == 'BA.1') | (
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.9') | (
                                                   Variant_in_week == 'BA.2.12.1') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((
                                                 Variant_in_week == 'AY.103') | (
                                                 Variant_in_week == 'AY.44') | (
                                                 Variant_in_week == 'AY.43') | (Variant_in_week == 'B.1.617.2') | (
                                                 Variant_in_week == 'AY.4') | (
                                                 Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.177') | (
                                                 Variant_in_week == 'B.1.2') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_78 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_78)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # dalla 107 alla 111
        if k == 111:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l] 
                    i_anomalie = np.where(((
                                                 Variant_in_week == 'BA.2.9') |(Variant_in_week == 'BA.2') |(
                                                   Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.12.1') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((
                                                 Variant_in_week == 'BA.1') | (
                                                 Variant_in_week == 'AY.103') | (
                                                 Variant_in_week == 'AY.44') | (
                                                 Variant_in_week == 'AY.43') | (Variant_in_week == 'B.1.617.2') | (
                                                 Variant_in_week == 'AY.4') | (
                                                 Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.177') | (
                                                 Variant_in_week == 'B.1.2') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_90 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_90)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # dalla 111 alla 121
        if k == 121:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((
                                                 Variant_in_week == 'BA.2.3') | (
                                                   Variant_in_week == 'BA.2.12.1') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((
                                                 Variant_in_week == 'BA.2.9') | (Variant_in_week == 'BA.2') | (
                                                 Variant_in_week == 'BA.1') | (
                                                 Variant_in_week == 'AY.103') | (
                                                 Variant_in_week == 'AY.44') | (
                                                 Variant_in_week == 'AY.43') | (Variant_in_week == 'B.1.617.2') | (
                                                 Variant_in_week == 'AY.4') | (
                                                 Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.177') | (
                                                 Variant_in_week == 'B.1.2') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_75 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_75)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # da 121 a 126
        if k == 126:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l] 
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((
                                                 Variant_in_week == 'BA.2.12.1') | (Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((
                                                 Variant_in_week == 'BA.2.9') | (
                                                 Variant_in_week == 'BA.2.3') | (Variant_in_week == 'BA.2') | (
                                                 Variant_in_week == 'BA.1.1') | (
                                                 Variant_in_week == 'AY.103') | (
                                                 Variant_in_week == 'AY.3') | (Variant_in_week == 'AY.25') | (
                                                 Variant_in_week == 'AY.44') | (Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.429') | (Variant_in_week == 'B.1.243') | (
                                                 Variant_in_week == 'B.1.240') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_75 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_75)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # da 126 a 134
        if k == 134:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((Variant_in_week == 'BA.5.1') | (
                                                   Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((
                                                 Variant_in_week == 'BA.2.12.1') | (
                                                 Variant_in_week == 'BA.2.9') | (
                                                 Variant_in_week == 'BA.2.3') | (Variant_in_week == 'BA.2') | (
                                                 Variant_in_week == 'BA.1.1') | (
                                                 Variant_in_week == 'AY.103') | (
                                                 Variant_in_week == 'AY.3') | (Variant_in_week == 'AY.25') | (
                                                 Variant_in_week == 'AY.44') | (Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.429') | (Variant_in_week == 'B.1.243') | (
                                                 Variant_in_week == 'B.1.240') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_75 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_75)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # dalla 134 a 156
        if k == 156:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((Variant_in_week == 'CH.1.1') | (Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((Variant_in_week == 'BA.5.1') | (
                                                 Variant_in_week == 'BA.2.12.1') | (
                                                 Variant_in_week == 'BA.2.9') | (
                                                 Variant_in_week == 'BA.2.3') | (Variant_in_week == 'BA.2') | (
                                                 Variant_in_week == 'BA.1.1') | (
                                                 Variant_in_week == 'AY.103') | (
                                                 Variant_in_week == 'AY.3') | (Variant_in_week == 'AY.25') | (
                                                 Variant_in_week == 'AY.44') | (Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.429') | (Variant_in_week == 'B.1.243') | (
                                                 Variant_in_week == 'B.1.240') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_75 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_75)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # DALLA 156 ALLA 159
        if k == 159:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    i_anomalie = np.where(((Variant_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predetti_in_week[i_anomalie])  
                    FN_week = np.sum(Casi_in_week[i_anomalie] - Predetti_in_week[i_anomalie])
                    i_inlier = np.where(((Variant_in_week == 'CH.1.1') | (Variant_in_week == 'BA.5.1') | (
                                                 Variant_in_week == 'BA.2.12.1') | (
                                                 Variant_in_week == 'BA.2.9') | (
                                                 Variant_in_week == 'BA.2.3') | (Variant_in_week == 'BA.2') | (
                                                 Variant_in_week == 'BA.1.1') | (
                                                 Variant_in_week == 'AY.103') | (
                                                 Variant_in_week == 'AY.3') | (Variant_in_week == 'AY.25') | (
                                                 Variant_in_week == 'AY.44') | (Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.429') | (Variant_in_week == 'B.1.243') | (
                                                 Variant_in_week == 'B.1.240') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_75 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_75)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
        # dalla 159 alla 160
        if k == 160:
            try:
                for i in range(min(Settimane_in_week_range), max(Settimane_in_week_range) + 1):
                    i_l = np.where(Settimane_in_week_range == i)  
                    Variant_in_week = Variant_in_week_range[i_l]  
                    Casi_in_week = Casi_in_week_range[i_l]  
                    Predetti_in_week = Predetti_in_week_range[i_l]  
                    #i_anomalie = np.where(())
                    TP_week = 0  
                    FN_week = 0
                    i_inlier = np.where(((Variant_in_week == 'XBB.1.5') | (Variant_in_week == 'CH.1.1') | (Variant_in_week == 'BA.5') | (
                                                 Variant_in_week == 'BA.2.12.1') | (
                                                 Variant_in_week == 'BA.2.9') | (
                                                 Variant_in_week == 'BA.2.3') | (Variant_in_week == 'BA.2') | (
                                                 Variant_in_week == 'BA.1.1') | (
                                                 Variant_in_week == 'AY.103') | (
                                                 Variant_in_week == 'AY.3') | (Variant_in_week == 'AY.25') | (
                                                 Variant_in_week == 'AY.44') | (Variant_in_week == 'B.1.1.7') | (
                                                 Variant_in_week == 'B.1.429') | (Variant_in_week == 'B.1.243') | (
                                                 Variant_in_week == 'B.1.240') | (Variant_in_week == 'B.1.1') | (
                                                 Variant_in_week == 'unknown') | (Variant_in_week == 'B.1')))
                    TN_week = np.sum(Casi_in_week[i_inlier] - Predetti_in_week[i_inlier])
                    FP_week = np.sum(Predetti_in_week[i_inlier])
                    FP_rate_75 = FP_week / (FP_week + TN_week + 0.001)
                    FP_RATE_FINAL.append(FP_rate_75)
                    TP_FINAL.append(TP_week)
                    FP_FINAL.append(FP_week)
                    FN_FINAL.append(FN_week)
                    TN_FINAL.append(TN_week)
                i = k
                continue
            except ValueError:
                print("The list is empty. Add values to the list")
                break
    Settimane_finali = np.unique(Settimane_giuste)  
    final_df = pd.DataFrame(FP_RATE_FINAL)
    sns.set(style="whitegrid")
    ax = sns.lineplot(data=final_df, color='#fde0dd')
    plt.bar((Settimane_finali - 2), FP_RATE_FINAL, color='#fa9fb5')
    # giving title to the plot
    plt.title('Fp_positive_rate')
    plt.xlabel('week')
    plt.ylabel('FP_RATE')
    plt.savefig(path_salvataggio + '/FP_in_time_completo.png')
    plt.close()

    # calcolo la precision
    precision = (np.array(TP_FINAL)) / (np.array(FP_FINAL) + np.array(TP_FINAL) + 0.001)

    plt.figure(figsize=(17, 8))
    plt.bar(Settimane_finali, precision, 0.4, color='#8856a7', alpha=0.7)
    plt.grid(color='#9ebcda', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    ax = plt.gca()
    ax.set_facecolor('#e0ecf4')
    for i in range(len(Settimane_finali)):
        if precision[i] > 0.01:
            plt.annotate(round(precision[i], 2), (Settimane_finali[i], precision[i]), size=14)
    plt.title('Precision')
    plt.xlabel("Weeks")
    plt.ylabel("Precision")
    plt.ylim(0.01, None)  # Imposta l'asse y per iniziare da 0.01
    plt.tight_layout()
    plt.savefig(path_salvataggio + '/precision_overall.png')
    # Calcolo matrici di confusione cumulate
    FP_SUM = np.cumsum(FP_FINAL)
    FN_SUM = np.cumsum(FN_FINAL)
    TP_SUM = np.cumsum(TP_FINAL)
    TN_SUM = np.cumsum(TN_FINAL)
    k = 'generale'
    for i in range(len(FN_SUM)):
         plot_confusion_matrix(TP_SUM[i], FP_SUM[i], TN_SUM[i], FN_SUM[i], k, Settimane_finali[i], path_salvataggio)

    return FP_RATE_FINAL, Settimane_finali, TN_FINAL, TP_FINAL, FP_FINAL, FN_FINAL


# path = ' '
# retraining_week = [27, 35, 45,48,49,51,62,75,90]
# # #retraining_week = [27,35]
# False_positive_rate,settimane_finali,TN_FINAL,TP_FINAL,FP_FINAL,FN_FINAL= falsepositive(measure_sensibilit, retraining_week, path)
# print(FP_FINAL)
# print(TP_FINAL)
