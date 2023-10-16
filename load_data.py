import pandas as pd
import os


def load_data(dir_dataset, week_range):
    week_range = [str(x) for x in week_range]
    weeks_folder = [x for x in os.listdir(dir_dataset) if x in week_range]
    df_list = []
    w_list = []
    for week in weeks_folder:
        df_path = dir_dataset  + week +'/week_dataset.txt'
        df = pd.read_csv(df_path, header=None)
        # df = df[~df.iloc[:, 0].isin(id_unknown)]
        df_list.append(df)
        w_list += [week]*df.shape[0]
        directory = os.path.join("c:\\", "path")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    f = open(file, 'r')
                    # quì eseguiamo i calcoli necessari
                    f.close()
    return pd.concat(df_list), w_list