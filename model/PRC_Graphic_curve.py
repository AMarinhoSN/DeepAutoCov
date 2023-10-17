import matplotlib.pyplot as plt

def calcola_prc(lista_grande,path_alvataggio):
    # initialization of sums for each of the 40 columns
    somme_precision = [0] * 40  # Una lista di 40 zeri
    somme_recall=[0]*40

    # Cycle through all sublists.
    for sottolista in lista_grande:
        # Cycle through all 40 columns (positions).
        for i in range(40):
            # I take the sub-sublist corresponding to column i
            sotto_sottolista = sottolista[i]

            # I add the value corresponding to "precision" (index 1) to the sum for column i
            somme_precision[i] += sotto_sottolista[1]
            somme_recall[i] += sotto_sottolista[2]

    # Printing sums
    for i, somma in enumerate(somme_precision):
        somme_precision[i] = somme_precision[i]/16
        somme_recall[i] = somme_recall[i]/16

    # Design PRC Curve
    plt.figure(1)
    plt.plot(somme_recall, somme_precision, '-', label='Autoencoder')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(str(path_alvataggio)+'/PRC.jpg', bbox_inches='tight')
    plt.show()
    info_graph='Il Grafico è stato stampato'

    return somme_precision,somme_recall,info_graph


