import pandas as pd
import matplotlib.pyplot as plt
def best_worst(path_save):
    # The Best
    data = {
        'lineage': ['BA.2', 'BA.1', 'BA.2.9', 'B.1.617.2', 'AY.43'],
        'week_first_seen': [42, 33, 39, 37, 48],
        'week_discovery': [57, 43, 39, 54, 58],
        'week_identified': [111, 111, 111, 87, 105],
        'week_5_percent': [107, 108, 109, 78, 82]
    }

    df = pd.DataFrame(data)

    # Create a copy of the original dataframe
    df_original = df.copy()

    # Normalization
    min_value = df[['week_first_seen', 'week_discovery', 'week_identified']].min(axis=1)
    df['week_first_seen'] -= min_value
    df['week_discovery'] -= min_value
    df['week_identified'] -= min_value
    df['week_5_percent'] -= min_value
    fig, ax = plt.subplots(figsize=(23, 12))

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

    for i, (row, row_original) in enumerate(zip(df.itertuples(), df_original.itertuples())):
        ax.plot(row.week_first_seen, i, marker='*', color='green', markersize=22,
                label='First seen in dataset' if i == 0 else "_nolegend_")
        ax.plot(row.week_discovery, i, marker='s', color='blue', markersize=22,
                label='Predicted by our model as DL' if i == 0 else "_nolegend_")
        ax.plot(row.week_identified, i, marker='o', color='red', markersize=22,
                label='Reach 25% dominance' if i == 0 else "_nolegend_")
        ax.plot(row.week_5_percent, i, marker='D', color='yellow', markersize=22,
                label='Reach 5% dominance' if i == 0 else "_nolegend_")
        ax.text(row.week_5_percent, i - 0.2, str(row_original.week_5_percent), color='black', ha='center', va='center',
                fontsize=20)

        ax.text(row.week_first_seen, i - 0.2, str(row_original.week_first_seen), color='black', ha='center',
                va='center', fontsize=20)
        ax.text(row.week_discovery, i - 0.2, str(row_original.week_discovery), color='black', ha='center', va='center',
                fontsize=20)
        ax.text(row.week_identified, i - 0.2, str(row_original.week_identified), color='black', ha='center',
                va='center', fontsize=20)

        if row.week_discovery != row.week_first_seen:
            ax.annotate("", xy=(row.week_discovery - 0.5, i), xytext=(row.week_first_seen + 0.5, i),
                        arrowprops=dict(arrowstyle="-", color='black', linewidth=3))
            week_diff_after = row_original.week_discovery - row_original.week_first_seen
            ax.text((row.week_discovery + row.week_first_seen) / 2, i + 0.15, str(week_diff_after), color='black',
                    ha='center', va='center', fontsize=20,
                    bbox=dict(facecolor='red', edgecolor='black', boxstyle='round,pad=0.2'))

        if row.week_identified != row.week_discovery:
            ax.annotate("", xy=(row.week_identified - 0.5, i), xytext=(row.week_discovery + 0.5, i),
                        arrowprops=dict(arrowstyle="->", color='black', linewidth=3))
            week_diff = row_original.week_identified - row_original.week_discovery
            ax.text((row.week_discovery + row.week_identified) / 2, i + 0.15, str(week_diff), color='black',
                    ha='center', va='center', fontsize=20,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, facecolor='#ffb6c1', alpha=0.5)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['lineage'], fontsize=20)

    # ax.set_xlabel('Week', fontsize=24)
    ax.set_ylabel('Lineage', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xticks([])

    ax.legend(fontsize=20)
    plt.title('World-Best Performance Lineages', fontsize=26)
    plt.savefig(str(path_save)+'/WorldPerformance.png', bbox_inches='tight')

    plt.show()

    # The Worst
    data = {
        'lineage': ['BA.2.12.1', 'B.1.2', 'AY.4', 'B.1.1.7', 'B.1.177'],
        'week_first_seen': [93, 4, 50, 22, 8],
        'week_discovery': [115, 11, 69, 30, 18],
        'week_identified': [126, 47, 79, 56, 44],
        'week_5_percent': [122, 33, 75, 51, 42]
    }

    df = pd.DataFrame(data)

    # Create a copy of the original dataframe
    df_original = df.copy()

    # Normalization
    min_value = df[['week_first_seen', 'week_discovery', 'week_identified']].min(axis=1)
    df['week_first_seen'] -= min_value
    df['week_discovery'] -= min_value
    df['week_identified'] -= min_value
    df['week_5_percent'] -= min_value

    fig, ax = plt.subplots(figsize=(23, 12))

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

    for i, (row, row_original) in enumerate(zip(df.itertuples(), df_original.itertuples())):
        ax.plot(row.week_first_seen, i, marker='*', color='green', markersize=22,
                label='First seen in dataset' if i == 0 else "_nolegend_")
        ax.plot(row.week_discovery, i, marker='s', color='blue', markersize=22,
                label='Predicted by our model as DL' if i == 0 else "_nolegend_")
        ax.plot(row.week_identified, i, marker='o', color='red', markersize=22,
                label='Reach 25% dominance' if i == 0 else "_nolegend_")
        ax.plot(row.week_5_percent, i, marker='D', color='yellow', markersize=22,
                label='Reach 5% dominance' if i == 0 else "_nolegend_")
        ax.text(row.week_5_percent, i - 0.2, str(row_original.week_5_percent), color='black', ha='center', va='center',
                fontsize=20)
        ax.text(row.week_first_seen, i - 0.2, str(row_original.week_first_seen), color='black', ha='center',
                va='center', fontsize=20)
        ax.text(row.week_discovery, i - 0.2, str(row_original.week_discovery), color='black', ha='center', va='center',
                fontsize=20)
        ax.text(row.week_identified, i - 0.2, str(row_original.week_identified), color='black', ha='center',
                va='center', fontsize=20)

        if row.week_discovery != row.week_first_seen:
            ax.annotate("", xy=(row.week_discovery - 0.2, i), xytext=(row.week_first_seen + 0.2, i),
                        arrowprops=dict(arrowstyle="-", color='black', linewidth=3))
            week_diff_after = row_original.week_discovery - row_original.week_first_seen
            ax.text((row.week_discovery + row.week_first_seen) / 2, i + 0.15, str(week_diff_after), color='black',
                    ha='center', va='center', fontsize=20,
                    bbox=dict(facecolor='red', edgecolor='black', boxstyle='round,pad=0.2'))

        if row.week_identified != row.week_discovery:
            ax.annotate("", xy=(row.week_identified - 0.2, i), xytext=(row.week_discovery + 0.2, i),
                        arrowprops=dict(arrowstyle="->", color='black', linewidth=3))
            week_diff = row_original.week_identified - row_original.week_discovery
            ax.text((row.week_discovery + row.week_identified) / 2, i + 0.15, str(week_diff), color='black',
                    ha='center', va='center', fontsize=20,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, facecolor='#ffb6c1', alpha=0.5)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['lineage'], fontsize=20)

    # ax.set_xlabel('Week', fontsize=24)
    ax.set_ylabel('Lineage', fontsize=24)
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xticks([])

    ax.legend(fontsize=20)
    plt.title('World-Worse Performance Lineages', fontsize=26)
    plt.savefig(str(path_save)+'/WorldPerformanceworst.png', bbox_inches='tight')

    plt.show()

    return()
