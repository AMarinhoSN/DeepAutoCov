import matplotlib.pyplot as plt

def measure_of_variants(TP,FP,TN,FN,k,week,path_save_file):
    #TP=true positive,FP=false positive, TN= True Negative, FN=false negative, k=variante, week=settimana giusta
    precision=TP/(TP+FP+0.0001)
    recall=TP/(TP+FN+0.0001)
    specificity=TN/(TN+FP+0.0001)
    # Grafici

    #precision
    plt.figure(figsize=(17, 8))
    plt.bar(week, precision, 0.4, color='#fde0dd', alpha=0.7)
    ax = plt.gca()
    ax.set_facecolor('#bcbddc')
    j = 2
    for i in range(len(week)):
        plt.annotate(round(precision[i],2), (week[i] , precision[i]))
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    plt.title('precision : ' + k)
    plt.xlabel("Week")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(path_save_file+'prec_in_time_' + k + '.png')
    plt.show()
    # recall
    plt.figure(figsize=(17, 8))
    plt.bar(week, recall, 0.4, color='#fde0dd', alpha=0.7)
    ax = plt.gca()
    ax.set_facecolor('#bcbddc')
    j = 2
    for i in range(len(week)):
        plt.annotate(round(recall[i],2), (week[i], recall[i]))
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    plt.title('Recall: ' + k)
    plt.xlabel("Week")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(path_save_file + 'rec_in_time_' + k + '.png')
    plt.show()
    #specificity
    plt.figure(figsize=(17, 8))
    plt.bar(week, specificity, 0.4, color='#fde0dd', alpha=0.7)
    ax = plt.gca()
    ax.set_facecolor('#bcbddc')
    j = 2
    for i in range(len(week)):
        plt.annotate(round(specificity[i],2), (week[i], specificity[i]))
    plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
    plt.title('Specificity: ' + k)
    plt.xlabel("Week")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(path_save_file + '/spec_in_time_' + k + '.png')
    plt.show()

    save=['Save the file']
    return(save)
