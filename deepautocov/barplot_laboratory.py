import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns

# graphic_fraction: graphic precision on top 100
# INPUT
#    1) AA: list that for each week of simulation [[FP,TP,N],[FP,TP,N]]
#    2) N: 100
#    3) path_salvataggio_file:path to save the image
# OUTPUT
#    1) image

def graphic_fraction(AA,N,path_salvataggio_file):
    x=[]

    AA_np=np.array(AA)


    TP_AA_np=np.array(list(map(int, AA_np[:, 1])))
    TOT_AA_np=np.array(list(map(int, AA_np[:, 2])))

    Fraction_AA=TP_AA_np/(TOT_AA_np +0.001)

    l=1
    for i in range(2,len(TOT_AA_np)+2):
        l=l+1
        x.append(l)

    x=np.array(x)
    width=0.25
    fig, ax = plt.subplots(figsize=(25, 8))
    rects2 = ax.bar(x, np.round(Fraction_AA,2), width / 2)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction TP/N')
    ax.set_xticks(x)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size= 8)

    #autolabel(rects1)
    autolabel(rects2)
    #autolabel(rects3)

    fig.tight_layout()
    plt.savefig(path_salvataggio_file + '/Fraction_general' + str(N) + '.png')


    k='Save'
    return k

# k=graphic_fraction(AA_100,100)
# print(k)
