import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from plot_confusion_matrix import *
from measure_of_variants import *
Specivit=[]
def sensitivity(measure_sensitivity, save_path):
    """
    This function calculates and plots sensitivity, specificity, and balanced accuracy for different lineages over time.
    
    Inputs:
    - measure_sensitivity: A list containing [['name_of_lineage', total_sequences, predicted_anomalies, week]] where total_sequences is the number of sequences in the week, and predicted_anomalies is the number of sequences predicted as anomalies in that week.
    - save_path: Path to save the output files.

    Output:
    - A list indicating the status of file saving and the saved file's information.
    """

    global Specificity
    measure_sensitivity_np = np.array(measure_sensitivity)
    Variants = measure_sensitivity_np[:, 0]
    variant_dict = Counter(Variants)
    
    for k in variant_dict.keys():
        if k == 'unknown':
            i_k = np.where(measure_sensitivity_np == k)[0]
            true = np.array(list(map(int, measure_sensitivity_np[i_k, 1])))
            FP = np.array(list(map(int, measure_sensitivity_np[i_k, 2])))
            week_an = np.array(list(map(int, measure_sensitivity_np[i_k, 3])))
            TN = true - FP
            Specificity = list(TN / (TN + FP + 0.000000001))
            week_pl = week_an + 1
            print(Specificity)
            plt.figure()
            plt.bar(week_pl, Specificity, color='royalblue', alpha=0.7)
            plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
            plt.xlabel("Week")
            plt.ylabel("Value")
            plt.title('Specificity of: ' + k)
            plt.savefig(save_path + 'Spec_in_time_' + k + '.png')
            continue

        # Process for non-unknown variants
        i_k = np.where(measure_sensitivity_np == k)[0]
        true = np.array(list(map(int, measure_sensitivity_np[i_k, 1])))
        predicted = np.array(list(map(int, measure_sensitivity_np[i_k, 2])))
        week_an = np.array(list(map(int, measure_sensitivity_np[i_k, 3])))
        anomaly_indices = week_an - 1  # assuming unknowns are always present
        Spec = np.array(Specificity)[anomaly_indices.astype(int)]
        FN = true - predicted
        i_no_neg = np.where(FN < 0)[0]
        FN[i_no_neg] = 0
        TP_cumulative = np.cumsum(predicted)
        FN_cumulative = np.cumsum(FN)
        TN_cumulative = np.cumsum(TN[anomaly_indices])
        FP_cumulative = np.cumsum(FP[anomaly_indices])
        df_conf = pd.DataFrame()
        df_conf['TN'] = TN_cumulative
        df_conf['FP'] = FP_cumulative
        df_conf['FN'] = FN_cumulative
        df_conf['TP'] = TP_cumulative
        df_conf.to_csv(save_path + 'Contingency_map_of_variant' + k + '.tsv', sep='\t', index=None)
        correct_week = week_an + 1
        for i in range(len(FN_cumulative)):
            plot_confusion_matrix(TP_cumulative[i], FP_cumulative[i], TN_cumulative[i], FN_cumulative[i], k, correct_week[i], save_path)

        save_info = measure_of_variants(predicted, FP[anomaly_indices], TN[anomaly_indices], FN, k, correct_week, save_path)
        Sensitivity = list(predicted / (predicted + FN + 0.000000001))
        Balanced_accuracy = (Spec + Sensitivity) / 2
        week_pl = correct_week
        print(Sensitivity)
        plt.figure(figsize=(17, 8))
        plt.bar(week_pl - 0.3, Sensitivity, 0.4, color='#fde0dd', alpha=0.7, label='Sensitivity')
        plt.bar(week_pl, Spec, 0.3, color='#fa9fb5', alpha=0.7, label='Specificity')
        plt.bar(week_pl + 0.3, Balanced_accuracy, 0.4, color='#c51b8a', alpha=0.7, label='Balanced_accuracy')
        ax = plt.gca()
        ax.set_facecolor('#bcbddc')
        for i in range(len(week_pl)):
            plt.annotate(true[i], (week_pl[i] - 0.4, Sensitivity[i]))
        plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
        plt.title('Measure of variants: ' + k)
        plt.legend(['Sensitivity', 'Specificity', 'Balanced Accuracy'])
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xlabel("Week")
        plt.ylabel("Value")
        plt.xticks(week_pl)
        plt.tight_layout()
        plt.savefig(save_path + 'sens_in_time_' + k + '.png')

    k = ["Save the file", save_info]
    return k
