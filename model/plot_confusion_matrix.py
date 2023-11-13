import matplotlib.pyplot as plt


def plot_confusion_matrix(tp, fp, tn, fn, k, week, path_salvataggio):
  """
  This function plots and saves a confusion matrix.
  Parameters:
  - tp: Number of True Positives.
  - fp: Number of False Positives.
  - tn: Number of True Negatives.
  - fn: Number of False Negatives.
  - k: Identifier for the lineage.
  - week: The week for which the confusion matrix is being plotted.
  - path_salvataggio: Path to save the plotted confusion matrix image.

  The function creates a matrix plot for the confusion matrix and saves it as an image file.
  """

  # Labels for the axes of the confusion matrix.
  labels = ['Positive', 'Negative']

  # Construct the confusion matrix.
  cm = [[tp, fn], [fp, tn]]

  # Initialize the figure and axes for the plot.
  fig = plt.figure()
  ax = fig.add_subplot(111)

  # Plot the confusion matrix using a specific color map.
  ax.matshow(cm, cmap='Pastel1')

  # Set the title and axis labels.
  plt.title('Confusion Matrix for Lineage ' + k + ' of week ' + str(week))
  plt.ylabel('True')
  plt.xlabel('Prediction')

  # Set ticks explicitly for both axes.
  ax.set_yticks([0, 1])
  ax.set_xticks([0, 1])

  # Set labels for the ticks.
  ax.set_yticklabels(labels)
  ax.set_xticklabels(labels)

  # Loop through the matrix and add text annotations for each cell.
  for i in range(2):
    for j in range(2):
      ax.text(j, i, cm[i][j], ha='center', va='center', color='black')

  # Save the confusion matrix plot to the specified path.
  plt.savefig(path_salvataggio + '/Confusion_matrix_image_of_variant' + k + '_week' + str(week) + '.png')

  # Close the plot to free up memory.
  plt.close()




# https://matplotlib.org/stable/tutorials/colors/colormaps.html per i colori di colormap