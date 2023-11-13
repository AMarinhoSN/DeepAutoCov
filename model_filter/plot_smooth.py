import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd

def plot_sma(vector, window_size,path_save):
    """
    It calculates the Simple Moving Average (SMA) of a vector and plots it together with the barplot of the vector.
    The window_size parameter indicates the size of the moving window.
    """
    sma = np.convolve(vector, np.ones(window_size) / window_size, mode='valid') #SMA  

    fig, ax1 = plt.subplots(figsize=(20, 12))  # Create a figure and a subplot.

    # Plot the bar graph
    ax1.bar(range(len(vector)), vector, 0.4, color='#66c2a5', alpha=0.7)
    ax1.plot(range(window_size - 1, len(vettore)), sma, 'r')
    ax1.set_title(str('False positive rate'), fontsize=26)
    #ax1.grid(False)  # Remove grid lines

    ax1.set_xlabel('Week', fontsize=24)  # Set x-axis label
    ax1.set_ylabel('False Positive Rate', fontsize=24)  # Set y-axis label
    ax1.tick_params(axis='both', which='major', labelsize=24)  # Set tick label size

    # Create an inset axes for the boxplot
    ax2 = inset_axes(ax1, width="40%", height="30%", loc='upper center')
    data_fp = {"False positive rate": vettore}
    df_fp = pd.DataFrame(data_fp)
    sns.boxplot(x="False positive rate", data=df_fp, ax=ax2)
    ax2.set_xlabel("FPR", fontsize=22)
    ax2.tick_params(axis='x', labelsize=22)  # Increase x-axis label size for boxplot
    ax2.grid(False)  # Remove grid lines
    plt.savefig(str(path_save)+'/FalsePositiveRate.png', bbox_inches='tight')
    plt.show()

