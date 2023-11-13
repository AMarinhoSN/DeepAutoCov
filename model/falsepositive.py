import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_confusion_matrix import *

def falsepositive(summary, retraining_week, path_to_save):
    
    summary_np = np.array(summary)
    Lineges = summary_np[:, 0]
    Number_of_case = np.array(list(map(int, summary_np[:, 1])))
    Predicted = np.array(list(map(int, summary_np[:, 2])))
    Week = np.array(list(map(int, summary_np[:, 3])))
    Correct_week = Week + 1
    variant_dict = Counter(Lineges)
    i = 0
    FP_RATE_FINAL = []
    Final = []
    TP_FINAL = []
    TN_FINAL = []
    FP_FINAL = []
    FN_FINAL = []

    for k in retraining_week:
        i_k = np.where(((Correct_week >= i) & (Correct_week < k)))
        Lineages_in_week_range = Lineges[i_k]
        Number_in_week_range = Number_of_case[i_k]
        Predicted_in_week_range = Predicted[i_k]
        week_in_week_range = Correct_week[i_k]
        # from 2 to 10
        if k == 10:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(
                        Lineages_in_week != 'unknown')
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(Lineages_in_week == 'unknown')
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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

        # from 10 to 11
        if k == 11:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((Lineages_in_week == 'B.1') | (
                            Lineages_in_week == 'B.1.177') | (Lineages_in_week == 'B.1.1.7') | (
                                                   Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'AY.44') | (
                                                   Lineages_in_week == 'AY.43') | (Lineages_in_week == 'AY.4') | (
                                                   Lineages_in_week == 'AY.103') | (Lineages_in_week == 'B.1.617.2') | (
                                                   Lineages_in_week == 'BA.1') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (Lineages_in_week == 'BA.2.12.1') | (
                                                   Lineages_in_week == 'BA.2') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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

        # from 11 to 44
        if k == 44:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((
                                                   Lineages_in_week == 'B.1.177') | (Lineages_in_week == 'B.1.1.7') | (
                                                   Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'AY.44') | (
                                                   Lineages_in_week == 'AY.43') | (Lineages_in_week == 'AY.4') | (
                                                   Lineages_in_week == 'AY.103') | (Lineages_in_week == 'B.1.617.2') | (
                                                   Lineages_in_week == 'BA.1') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (
                                                   Lineages_in_week == 'BA.2') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((Lineages_in_week == 'B.1.1') | (Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 44 to 47
        if k == 47:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((
                                                   Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'B.1.1.7') | (Lineages_in_week == 'AY.44') | (
                                                   Lineages_in_week == 'AY.43') | (Lineages_in_week == 'AY.4') | (
                                                   Lineages_in_week == 'AY.103') | (Lineages_in_week == 'B.1.617.2') | (
                                                   Lineages_in_week == 'BA.1') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (
                                                   Lineages_in_week == 'BA.2') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((
                                                 Lineages_in_week == 'B.1.177') | (Lineages_in_week == 'B.1.1') | (Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 47 to 56
        if k == 56:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((Lineages_in_week == 'B.1.1.7') | (
                            Lineages_in_week == 'AY.44') | (
                                                   Lineages_in_week == 'AY.43') | (Lineages_in_week == 'AY.4') | (
                                                   Lineages_in_week == 'AY.103') | (Lineages_in_week == 'B.1.617.2') | (
                                                   Lineages_in_week == 'BA.1') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (
                                                   Lineages_in_week == 'BA.2') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((
                                                 Lineages_in_week == 'B.1.177') | (
                                                 Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 56 to 79
        if k == 79:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((
                                                   Lineages_in_week == 'AY.44') | (
                                                   Lineages_in_week == 'AY.43') | (Lineages_in_week == 'AY.4') | (
                                                   Lineages_in_week == 'AY.103') | (Lineages_in_week == 'B.1.617.2') | (
                                                   Lineages_in_week == 'BA.1') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (
                                                   Lineages_in_week == 'BA.2') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((Lineages_in_week == 'B.1.1.7') | (
                            Lineages_in_week == 'B.1.177') | (
                                                 Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 79 to 82
        if k == 82:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((
                                                   Lineages_in_week == 'AY.44') | (
                                                   Lineages_in_week == 'AY.43') | (
                                                   Lineages_in_week == 'AY.103') | (Lineages_in_week == 'B.1.617.2') | (
                                                   Lineages_in_week == 'BA.1') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (
                                                   Lineages_in_week == 'BA.2') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((Lineages_in_week == 'AY.4') | (Lineages_in_week == 'B.1.1.7') | (
                            Lineages_in_week == 'B.1.177') | (
                                                 Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 82 to 84
        if k == 84:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((Lineages_in_week == 'B.1.617.2') | (
                            Lineages_in_week == 'AY.43') | (
                                                   Lineages_in_week == 'AY.103') | (
                                                   Lineages_in_week == 'BA.1') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (
                                                   Lineages_in_week == 'BA.2') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((
                                                 Lineages_in_week == 'AY.44') | (Lineages_in_week == 'AY.4') | (Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.177') | (
                                                 Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 84 to 87
        if k == 87:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((Lineages_in_week == 'B.1.617.2') | (
                            Lineages_in_week == 'AY.43') | (
                                                   Lineages_in_week == 'BA.1') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (
                                                   Lineages_in_week == 'BA.2') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((
                                                 Lineages_in_week == 'AY.44') | (
                                                 Lineages_in_week == 'AY.103') | (Lineages_in_week == 'AY.4') | (
                                                 Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.177') | (
                                                 Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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

        # from 87 to 105
        if k == 105:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((
                                                   Lineages_in_week == 'AY.43') | (
                                                   Lineages_in_week == 'BA.1') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (
                                                   Lineages_in_week == 'BA.2') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((
                                                 Lineages_in_week == 'AY.103') | (
                                                 Lineages_in_week == 'AY.44') | (Lineages_in_week == 'B.1.617.2') | (
                                                 Lineages_in_week == 'AY.4') | (
                                                 Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.177') | (
                                                 Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 105 to 107
        if k == 107:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((Lineages_in_week == 'BA.2') | (
                            Lineages_in_week == 'BA.1') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.9') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((
                                                 Lineages_in_week == 'AY.103') | (
                                                 Lineages_in_week == 'AY.44') | (
                                                 Lineages_in_week == 'AY.43') | (Lineages_in_week == 'B.1.617.2') | (
                                                 Lineages_in_week == 'AY.4') | (
                                                 Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.177') | (
                                                 Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 107 to 111
        if k == 111:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((
                                                   Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((
                                                 Lineages_in_week == 'BA.1') | (
                                                 Lineages_in_week == 'AY.103') | (
                                                 Lineages_in_week == 'AY.44') | (
                                                 Lineages_in_week == 'AY.43') | (Lineages_in_week == 'B.1.617.2') | (
                                                 Lineages_in_week == 'AY.4') | (
                                                 Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.177') | (
                                                 Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 111 to 121
        if k == 121:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((
                                                   Lineages_in_week == 'BA.2.3') | (
                                                   Lineages_in_week == 'BA.2.12.1') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((
                                                 Lineages_in_week == 'BA.2.9') | (Lineages_in_week == 'BA.2') | (
                                                 Lineages_in_week == 'BA.1') | (
                                                 Lineages_in_week == 'AY.103') | (
                                                 Lineages_in_week == 'AY.44') | (
                                                 Lineages_in_week == 'AY.43') | (Lineages_in_week == 'B.1.617.2') | (
                                                 Lineages_in_week == 'AY.4') | (
                                                 Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.177') | (
                                                 Lineages_in_week == 'B.1.2') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 121 to 126
        if k == 126:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((
                                                   Lineages_in_week == 'BA.2.12.1') | (Lineages_in_week == 'BA.5.1') | (
                                                   Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((
                                                 Lineages_in_week == 'BA.2.9') | (
                                                 Lineages_in_week == 'BA.2.3') | (Lineages_in_week == 'BA.2') | (
                                                 Lineages_in_week == 'BA.1.1') | (
                                                 Lineages_in_week == 'AY.103') | (
                                                 Lineages_in_week == 'AY.3') | (Lineages_in_week == 'AY.25') | (
                                                 Lineages_in_week == 'AY.44') | (Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.429') | (Lineages_in_week == 'B.1.243') | (
                                                 Lineages_in_week == 'B.1.240') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 126 to 134
        if k == 134:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((Lineages_in_week == 'BA.5.1') | (
                            Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((
                                                 Lineages_in_week == 'BA.2.12.1') | (
                                                 Lineages_in_week == 'BA.2.9') | (
                                                 Lineages_in_week == 'BA.2.3') | (Lineages_in_week == 'BA.2') | (
                                                 Lineages_in_week == 'BA.1.1') | (
                                                 Lineages_in_week == 'AY.103') | (
                                                 Lineages_in_week == 'AY.3') | (Lineages_in_week == 'AY.25') | (
                                                 Lineages_in_week == 'AY.44') | (Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.429') | (Lineages_in_week == 'B.1.243') | (
                                                 Lineages_in_week == 'B.1.240') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 134 to 156
        if k == 156:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((Lineages_in_week == 'BA.5.1') | (
                            Lineages_in_week == 'BA.2.12.1') | (
                                                 Lineages_in_week == 'BA.2.9') | (
                                                 Lineages_in_week == 'BA.2.3') | (Lineages_in_week == 'BA.2') | (
                                                 Lineages_in_week == 'BA.1.1') | (
                                                 Lineages_in_week == 'AY.103') | (
                                                 Lineages_in_week == 'AY.3') | (Lineages_in_week == 'AY.25') | (
                                                 Lineages_in_week == 'AY.44') | (Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.429') | (Lineages_in_week == 'B.1.243') | (
                                                 Lineages_in_week == 'B.1.240') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 156 to 159
        if k == 159:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    i_anomaly = np.where(((Lineages_in_week == 'XBB.1.5')))
                    TP_week = np.sum(Predicted_in_week[i_anomaly])
                    FN_week = np.sum(number_in_week[i_anomaly] - Predicted_in_week[i_anomaly])
                    i_inlier = np.where(((Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'BA.5.1') | (
                            Lineages_in_week == 'BA.2.12.1') | (
                                                 Lineages_in_week == 'BA.2.9') | (
                                                 Lineages_in_week == 'BA.2.3') | (Lineages_in_week == 'BA.2') | (
                                                 Lineages_in_week == 'BA.1.1') | (
                                                 Lineages_in_week == 'AY.103') | (
                                                 Lineages_in_week == 'AY.3') | (Lineages_in_week == 'AY.25') | (
                                                 Lineages_in_week == 'AY.44') | (Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.429') | (Lineages_in_week == 'B.1.243') | (
                                                 Lineages_in_week == 'B.1.240') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
        # from 159 to 160
        if k == 160:
            try:
                for i in range(min(week_in_week_range), max(week_in_week_range) + 1):
                    i_l = np.where(week_in_week_range == i)
                    Lineages_in_week = Lineages_in_week_range[i_l]
                    number_in_week = Number_in_week_range[i_l]
                    Predicted_in_week = Predicted_in_week_range[i_l]
                    #i_anomalie = np.where(())
                    TP_week = 0  
                    FN_week = 0
                    i_inlier = np.where(((Lineages_in_week == 'XBB.1.5') | (Lineages_in_week == 'CH.1.1') | (Lineages_in_week == 'BA.5') | (
                            Lineages_in_week == 'BA.2.12.1') | (
                                                 Lineages_in_week == 'BA.2.9') | (
                                                 Lineages_in_week == 'BA.2.3') | (Lineages_in_week == 'BA.2') | (
                                                 Lineages_in_week == 'BA.1.1') | (
                                                 Lineages_in_week == 'AY.103') | (
                                                 Lineages_in_week == 'AY.3') | (Lineages_in_week == 'AY.25') | (
                                                 Lineages_in_week == 'AY.44') | (Lineages_in_week == 'B.1.1.7') | (
                                                 Lineages_in_week == 'B.1.429') | (Lineages_in_week == 'B.1.243') | (
                                                 Lineages_in_week == 'B.1.240') | (Lineages_in_week == 'B.1.1') | (
                                                 Lineages_in_week == 'unknown') | (Lineages_in_week == 'B.1')))
                    TN_week = np.sum(number_in_week[i_inlier] - Predicted_in_week[i_inlier])
                    FP_week = np.sum(Predicted_in_week[i_inlier])
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
    final_week = np.unique(Correct_week)
    final_df = pd.DataFrame(FP_RATE_FINAL)
    sns.set(style="whitegrid")
    ax = sns.lineplot(data=final_df, color='#fde0dd')
    plt.bar((final_week - 2), FP_RATE_FINAL, color='#fa9fb5')
    # giving title to the plot
    plt.title('Fp_positive_rate')
    plt.xlabel('week')
    plt.ylabel('FP_RATE')
    plt.savefig(path_to_save + '/FP_in_time_completo.png')
    plt.close()

    precision = (np.array(TP_FINAL)) / (np.array(FP_FINAL) + np.array(TP_FINAL) + 0.001)

    plt.figure(figsize=(17, 8))
    plt.bar(final_week, precision, 0.4, color='#8856a7', alpha=0.7)
    plt.grid(color='#9ebcda', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    ax = plt.gca()
    ax.set_facecolor('#e0ecf4')
    for i in range(len(final_week)):
        if precision[i] > 0.01:
            plt.annotate(round(precision[i], 2), (final_week[i], precision[i]), size=14)
    plt.title('Precision')
    plt.xlabel("Weeks")
    plt.ylabel("Precision")
    plt.ylim(0.01, None)
    plt.tight_layout()
    plt.savefig(path_to_save + '/precision_overall.png')

    FP_SUM = np.cumsum(FP_FINAL)
    FN_SUM = np.cumsum(FN_FINAL)
    TP_SUM = np.cumsum(TP_FINAL)
    TN_SUM = np.cumsum(TN_FINAL)
    k = 'generale'
    for i in range(len(FN_SUM)):
         plot_confusion_matrix(TP_SUM[i], FP_SUM[i], TN_SUM[i], FN_SUM[i], k, final_week[i], path_to_save)

    return FP_RATE_FINAL, final_week, TN_FINAL, TP_FINAL, FP_FINAL, FN_FINAL


# path = ' '
# retraining_week = [27, 35, 45,48,49,51,62,75,90]
# # #retraining_week = [27,35]
# False_positive_rate,settimane_finali,TN_FINAL,TP_FINAL,FP_FINAL,FN_FINAL= falsepositive(measure_sensibilit, retraining_week, path)
# print(FP_FINAL)
# print(TP_FINAL)
