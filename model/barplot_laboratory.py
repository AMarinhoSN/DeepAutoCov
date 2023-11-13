import numpy as np
import matplotlib.pyplot as plt

def graphic_fraction(AA, N, path_salvataggio_file):
    # This function creates a bar chart representing the fraction of True Positives (TP)
    # over the total count (N) for different amino acids (AA).
    # Parameters:
    # AA: A list or array containing the amino acid data.
    # N: A numerical identifier or count.
    # path_salvataggio_file: The file path where the graph image will be saved.

    # Initialize an empty list to store x-axis values.
    x = []

    # Convert AA to a NumPy array for easier manipulation.
    AA_np = np.array(AA)

    # Extract True Positive counts and Total counts from AA_np, converting them to integers.
    TP_AA_np = np.array(list(map(int, AA_np[:, 1])))
    TOT_AA_np = np.array(list(map(int, AA_np[:, 2])))

    # Calculate the fraction of TP over Total for each amino acid, adding a small value to avoid division by zero.
    Fraction_AA = TP_AA_np / (TOT_AA_np + 0.001)

    # Generate x-axis values.
    l = 1
    for i in range(2, len(TOT_AA_np) + 2):
        l = l + 1
        x.append(l)

    # Convert the x-axis list to a NumPy array.
    x = np.array(x)

    # Set the width of each bar in the bar chart.
    width = 0.25

    # Create a subplot and define its size.
    fig, ax = plt.subplots(figsize=(25, 8))

    # Create a bar chart with Fraction_AA values.
    rects2 = ax.bar(x, np.round(Fraction_AA, 2), width / 2)

    # Set labels and titles for the axes and the chart.
    ax.set_ylabel('Fraction')
    ax.set_title('Fraction TP/N')
    ax.set_xticks(x)
    ax.legend()

    def autolabel(rects):
        # Internal function to label the bars with their respective height values.
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size=8)

    # Apply the label function to the bars.
    autolabel(rects2)

    # Adjust layout and save the figure to the specified path.
    fig.tight_layout()
    plt.savefig(path_salvataggio_file + '/Fraction_general' + str(N) + '.png')

    # Return a string indicating the saving action.
    k = 'Save'
    return k


