import matplotlib.pyplot as plt

def plot_confusion_matrix(tp, fp, tn, fn, k, week, path_salvataggio):
  labels = ['Positivo', 'Negativo']
  cm = [[tp, fn], [fp, tn]]

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.matshow(cm, cmap='Pastel1')
  plt.title('Matrice di confusione of variant ' + k + ' of week ' + str(week))
  plt.ylabel('Verità')
  plt.xlabel('Predizione')

  # Impostare le ticks in modo esplicito
  ax.set_yticks([0, 1])
  ax.set_xticks([0, 1])

  ax.set_yticklabels(labels)
  ax.set_xticklabels(labels)

  for i in range(2):
    for j in range(2):
      ax.text(j, i, cm[i][j], ha='center', va='center', color='black')

  plt.savefig(path_salvataggio + '/Confusion_matrix_image_of_variant' + k + '_week' + str(week) + '.png')
  plt.close()




# https://matplotlib.org/stable/tutorials/colors/colormaps.html per i colori di colormap