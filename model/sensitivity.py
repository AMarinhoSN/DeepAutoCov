import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from plot_confusion_matrix import *
from measure_of_variants import *
Specivit=[]
def sensitivity(measure_sensibilit,path_salvataggio_file):
    global Specivit
    measure_sensibilit_np = np.array(measure_sensibilit)
    Varianti = measure_sensibilit_np[:, 0]
    variant_dict = Counter(Varianti)
    for k in variant_dict.keys():
        if k == 'unknown':
            i_k = np.where(measure_sensibilit_np == k)[0]
            veri = np.array(list(map(int, measure_sensibilit_np[i_k, 1])))
            FP = np.array(list(map(int, measure_sensibilit_np[i_k, 2])))
            week_an = np.array(list(map(int, measure_sensibilit_np[i_k, 3])))
            TN = veri - FP
            Specivit = list(TN / (TN + FP + 0.000000001))
            week_pl = week_an + 1
            print(Specivit)
            plt.figure()
            plt.bar(week_pl, Specivit, color='royalblue', alpha=0.7)
            plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
            plt.xlabel("Week")
            plt.ylabel("Value")
            plt.title('Specificità di: ' + k)
            plt.savefig(path_salvataggio_file+'Spec_in_time_' + k + '.png')

            continue

        i_k = np.where(measure_sensibilit_np == k)[0]
        veri = np.array(list(map(int, measure_sensibilit_np[i_k, 1])))
        predetti = np.array(list(map(int, measure_sensibilit_np[i_k, 2])))
        week_an = np.array(list(map(int, measure_sensibilit_np[i_k, 3])))
        Indici_anomalie = week_an - 1  # si suppone che gli unkown sono sempre presenti
        Spec = np.array(Specivit)[Indici_anomalie.astype(int)]
        FN = veri - predetti
        i_no_neg = np.where(FN < 0)[0]
        FN[i_no_neg] = 0
        TP_an_cum = np.cumsum(predetti)
        FN_an_cum = np.cumsum(FN)
        TN_an_cum = np.cumsum(TN[Indici_anomalie])
        FP_an_cum = np.cumsum(FP[Indici_anomalie])
        df_conf = pd.DataFrame()
        df_conf['TN'] = TN_an_cum
        df_conf['FP'] = FP_an_cum
        df_conf['FN'] = FN_an_cum
        df_conf['TP'] = TP_an_cum
        df_conf.to_csv(path_salvataggio_file+'Contigency_map_of_variant' + k + '.tsv', sep='\t', index=None)
        settimana_giusta=week_an+1
        for i in range(len(FN_an_cum)):
            plot_confusion_matrix(TP_an_cum[i], FP_an_cum[i], TN_an_cum[i], FN_an_cum[i], k, settimana_giusta[i],path_salvataggio_file)

        salvataggio = measure_of_variants(predetti, FP[Indici_anomalie], TN[Indici_anomalie], FN, k, settimana_giusta,
                                          path_salvataggio_file)
        Sensitivit = list(predetti / (predetti + FN + 0.000000001))
        Balanced_accuracy = (Spec + Sensitivit) / 2
        week_pl = settimana_giusta
        print(Sensitivit)
        plt.figure(figsize=(17, 8))
        plt.bar(week_pl - 0.3, Sensitivit, 0.4, color='#fde0dd', alpha=0.7, label='Sensibilità')
        plt.bar(week_pl, Spec, 0.3, color='#fa9fb5', alpha=0.7, label='Specificità')
        plt.bar(week_pl + 0.3, Balanced_accuracy, 0.4, color='#c51b8a', alpha=0.7, label='Balanced_accuracy')
        ax = plt.gca()
        ax.set_facecolor('#bcbddc')
        j = 2
        for i in range(len(week_pl)):
            plt.annotate(veri[i], (week_pl[i] - 0.4, Sensitivit[i]))
        plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
        plt.title('Measure of variants: ' + k)
        plt.legend(['Sensibility', 'Specificity', 'Balanced Accuracy'])
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.xlabel("Week")
        plt.ylabel("Value")
        plt.xticks(week_pl)
        plt.tight_layout()
        plt.savefig(path_salvataggio_file+'sens_in_time_' + k + '.png')

    k = ["Save the file",salvataggio]
    return k

