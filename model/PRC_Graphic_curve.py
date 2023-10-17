import matplotlib.pyplot as plt

def calcola_prc(lista_grande,path_alvataggio):
    # Inizializzazione delle somme per ciascuna delle 40 colonne
    somme_precision = [0] * 40  # Una lista di 40 zeri
    somme_recall=[0]*40

    # Ciclo attraverso tutte le sottoliste
    for sottolista in lista_grande:
        # Ciclo attraverso tutte le 40 colonne (posizioni)
        for i in range(40):
            # Prendo la sotto-sottolista corrispondente alla colonna i
            sotto_sottolista = sottolista[i]

            # Sommo il valore corrispondente a "precision" (indice 1) alla somma per la colonna i
            somme_precision[i] += sotto_sottolista[1]
            somme_recall[i] += sotto_sottolista[2]

    # Stampa delle somme
    for i, somma in enumerate(somme_precision):
        somme_precision[i] = somme_precision[i]/16
        somme_recall[i] = somme_recall[i]/16

    # Disegnare la curva PRC originale
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


