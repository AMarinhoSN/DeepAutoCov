from load_data import *
from get_lineage_class import *
from map_lineage_to_finalclass import *
from optparse import OptionParser
from datetime import *
from Autoencoder_training_GPU import *
import logging
from sklearn.model_selection import ParameterGrid
from sensitivity import *
from fraction_mail_postfiltering import *
from barplot_laboratory import *
from scoperta import *
from filter_dataset import *
from test_normality_error import *
import gc
from model_dl import *
from Best_worse import *
from plot_smooth import *
from kmers_error import *
from Lookuptable import *
from Lineages_of_interest import *
from PRC_Graphic_curve import *
from PRC_curve import *

def main(options):
    # memory control
    gc.enable()
    # use GPU
    strategy = tf.distribute.MirroredStrategy()

    # define a list that we will use during the code
    beta = []  # for 100
    measure_sensibilit = []  # salvo i lineages e tutto
    NF = []  # sta per number feature per settimana
    results_fine_tune = []
    fractions_100 = []
    mse_prc_curve = []  # per le probabilità della regressione logistica
    true_class_prc_curve = []  # contiene la classe vera
    info_PCR = []  # conterrà tutto le curve create
    ind_prc = 0
    summary_100_anomalies = []
    summary_100_anomalies_percentage = []

    #dir_week = '/mnt/resources/2022_04/2022_04/dataset_week/'
    dir_week =str(options.path_drive) #path del Dataset #path del Dataset
    #metadata = pd.read_csv('/mnt/resources/2022_04/2022_04/filtered_metadata_0328_weeks.csv')
    metadata = pd.read_csv(str(options.csv_path)) # leggo il file che devo salavare dove creo il dtaaset
    metadata_2 = pd.read_csv(str(options.csv_path)) # leggo il file che devo salavare dove creo il dtaaset


    # columns in metadata
    col_class_lineage = 'Pango.lineage'
    col_submission_date = 'Collection.date'
    col_lineage_id = 'Accession.ID'

    # Prima sostituzione perche il replace è utile in casi semplici e non in quelli compessi
    valid_lineage, valid_lineage_prc, dizionario_lineage_settimane, lineages_know = lineages_of_interest()

    metadata[col_class_lineage] = metadata[col_class_lineage].apply(lambda x: 'unknown' if x not in valid_lineage else x) # metto gli unknow al posto degli altri lineage non considerati come di interersse
    id_unknown = metadata[metadata[col_class_lineage] == 'unknown'][col_lineage_id].tolist() # trovo gli id degli unknown

    # week of retraining
    retraining_week, retraining_week_false_positive = retraining_weeks()

    # K-mers
    header = pd.read_csv(str(options.kmers), nrows=1) # qua devo mettere il primo file della prima settimaa con l'header
    features = header.columns[1:].tolist()  # mi da i k-mers
    print('-----------------------------------------------------------------------------------')
    print('i k-mers in totale sono : ' + str(len(features)))
    print('-----------------------------------------------------------------------------------')

    # Saving documents
    path_salvataggio_file=str(options.path_save) #mettere il file in cui salviamo i grafici

    # lineage og interest
    lineage_of_interest = metadata[col_class_lineage].unique().tolist()
    lineage_of_interest.remove('unknown')


    # logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            # logging.FileHandler('/mnt/resources/2022_04/2022_04/'+'run_main_oneclass_retrain_tmp.log', 'w+'),
                            logging.FileHandler(path_salvataggio_file + '/Autoencode_performance.log', 'w+'),
                            logging.StreamHandler()
                        ])


    # Training week
    starting_week = 1 # qua posso mettere più valori perchè ne ho solo 17 nella prima settimana e non può andare bene

    # Loading first training step
    df_trainstep_1, train_w_list = load_data(dir_week, [starting_week])
    train_step1 = df_trainstep_1.iloc[:, 1:len(df_trainstep_1.columns)].to_numpy()

    # define the features mantain
    sum_train = np.sum(train_step1, axis=0)
    keepFeature=sum_train/len(train_step1)
    i_no_zero = np.where(keepFeature >= options.rate_mantain)[0] # tengo le feature che sono diverse di alamneo il N%

    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('il numero di feature delle sequenze diverso da zero sono :' + str((len(i_no_zero))))
    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')

    y_train_initial = metadata[metadata[col_lineage_id].isin(df_trainstep_1.iloc[:, 0].tolist())][col_class_lineage]
    y_train_class = map_lineage_to_finalclass(y_train_initial.tolist(), lineage_of_interest)
    counter_i = Counter(y_train_initial)  # at the beginning, all the lineages were "unknown"=neutral

    # filtering out features with all zero
    train_step_completo = train_step1
    train = train_step1[:, i_no_zero] # Per il primo allenamento metto il train
    lineages_train=np.array(y_train_initial.tolist()) # lineages

    tf.random.set_seed(10)
    # we Create the Autoencoder models

    # Setto i paramentri dell'autoencoder
    nb_epoch = options.number_epoch
    batch_size = options.batch_size #train_step1.shape[0]
    input_dim =train.shape[1] #num of columns
    encoding_dim = options.encoding_dim
    hidden_dim_1 = int(encoding_dim / 2) #512
    hidden_dim_2=int(hidden_dim_1/2) #256
    hidden_dim_3=int(hidden_dim_2/2) #128
    hidden_dim_4=int(hidden_dim_3/2) #64
    hidden_dim_5=int(hidden_dim_4/2) #32
    reduction_factor = options.red_factor

    p_grid = {'nb_epoch':[nb_epoch],'batch_size':[batch_size],'input_dim':[input_dim],'encoding_dim':[encoding_dim],'hidden_dim_1':[int(encoding_dim / 2)],'hidden_dim_2':[hidden_dim_2],'hidden_dim_3':[hidden_dim_3],'hidden_dim_4':[hidden_dim_4],'hidden_dim_5':[hidden_dim_5],'Reduction_factor':[reduction_factor]}
    all_combo = list(ParameterGrid(p_grid))

    with strategy.scope():
        autoencoder=model(input_dim,encoding_dim,hidden_dim_1,hidden_dim_2,hidden_dim_3,hidden_dim_4,hidden_dim_5,reduction_factor,path_salvataggio_file)
    for combo in all_combo[0:1]:
        combo
        logging.info("---> Autoencoder - Param: " + str(combo))
        y_test_dict_variant_type = {}
        y_test_dict_finalclass = {}
        y_test_dict_predictedclass = {}
        #train = train_step1.copy()
        history = autoencoder_training_GPU(autoencoder,train, train,nb_epoch,batch_size)
        print('Ho allenato la rete neurale : ')
        print(history)
        # CALCOLO P-VALUE
        info, mse_tr = test_normality(autoencoder, train) # we calultae the mse in a normal dataset
        for week in range(1, 159):  # metadata['week'].max()
            if week in retraining_week:
                logging.info('----> RETRAINING <-----')

                ind_prc = ind_prc + 1
                # soglia = threshold_fixed
                # min_prob = min(mse_prc_curve)
                # max_prob = max(mse_prc_curve)
                # print('--------PRC--------')
                # info_pcr = evaluate_pcr(true_class_prc_curve, mse_prc_curve, soglia, min_prob, max_prob)
                # info_PCR.append(info_pcr)

                # We create a new training set for retrain the network
                train_model_value = train_step_completo # soloziono solo i kmaers
                classi=lineages_train #seleziono solo i valori
                sum_train = np.sum(train_model_value, axis=0)
                keepFeature = sum_train / len(train_model_value)
                i_no_zero = np.where(keepFeature > options.rate_mantain)[0]

                number_feature = len(i_no_zero)
                print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('il numero di feature in almeno il 5% delle sequenze diverso da zero sono :' + str((len(i_no_zero))))
                print('---------------------------------------------------------------------------------------------------------------------------------------------------------')

                train_model_value = train_model_value[:, i_no_zero]

                # seleziono le righe di interesse
                index_raw = trova_indici_lineage_per_settimana(classi, week, dizionario_lineage_settimane) # prende in ingresso solo le classi
                train_model_value=train_model_value[index_raw,:]
                np.random.shuffle(train_model_value)
                NF.append(number_feature)

                batch_size=512
                input_dim =train_model_value.shape[1]
                with strategy.scope():
                    autoencoder=model(input_dim, encoding_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, hidden_dim_4, hidden_dim_5,
                          reduction_factor, path_salvataggio_file)
                history = autoencoder_training_GPU(autoencoder, train_model_value, train_model_value, nb_epoch, batch_size)

                print('Ho allenato la rete neurale : ')
                print(history)

                info,mse_tr = test_normality(autoencoder, train_model_value)
                train_model_value = []
                classi=[]
                mse_prc_curve = []
                true_class_prc_curve = []
            #week_date_list = list(set(metadata[metadata['week'] == starting_week + week][col_submission_date].tolist()))
            #week_date = str(min([get_time(x) for x in week_date_list]))
            logging.info("# Week " + str(starting_week + week))
            print("# Week " + str(starting_week + week))

            # Loading test step
            df_teststep_i, test_w_list = load_data(dir_week, [starting_week + week])
            test_step_i = df_teststep_i.iloc[:, 1:len(df_teststep_i.columns)].to_numpy() # transform in numpy
            id_identifier = df_teststep_i.iloc[:, 0].to_list()
            test_step_completo = test_step_i
            test_step_i = test_step_i[:, i_no_zero] # feature selections
            y_test_step_i = get_lineage_class(metadata, df_teststep_i.iloc[:, 0].tolist()) #Mi da gli id list
            lineages_l=metadata[metadata[col_lineage_id].isin(df_teststep_i.iloc[:, 0].tolist())][col_class_lineage]
            lineages_test=np.array(lineages_l.tolist()) # lineages in the week
            #test_with_class_completo = np.column_stack((test_step_completo, lineages))
            y_test_dict_variant_type[starting_week + week] = y_test_step_i
            y_test_fclass_i = map_lineage_to_finalclass(y_test_step_i, lineage_of_interest)  # lineage_of_interest
            y_test_fclass_i_prc = map_lineage_to_finalclass(y_test_step_i, valid_lineage_prc[ind_prc])
            i_voc = np.where(np.array(y_test_fclass_i) == -1)[0]
            y_test_dict_finalclass[starting_week + week] = y_test_fclass_i
            lineage_dict = Counter(y_test_step_i)
            test_x_predictions = autoencoder.predict(test_step_i) # total prediction

            # lineage name for prediction
            lineages_error = metadata_2[metadata_2[col_lineage_id].isin(df_teststep_i.iloc[:, 0].tolist())][col_class_lineage]
            lineages_error_test = np.array(lineages_error.tolist())

            # Treshold
            mse = np.mean(np.power(test_step_i - test_x_predictions, 2), axis=1)
            mse_prc=list(mse)
            mse_prc_curve += mse_prc  # prob
            true_class_prc_curve += y_test_fclass_i_prc  # classe vera
            error_df = pd.DataFrame({'Reconstruction_error': mse})
            threshold_fixed = np.mean(mse_tr) + 1.5 * np.std(mse_tr)

            print('la soglia è: ' + str(threshold_fixed))
            y_test_i_predict = [-1 if e >= threshold_fixed else 1 for e in error_df.Reconstruction_error.values]
            y_test_i_predict = np.array(y_test_i_predict)

            i_inlier = np.where(y_test_i_predict == 1)[
                0]  # qua seleziono solo i predetti inlier cioè quelli che il mio classificiatore predice come inlier

            # selection the first 100 with highest mse
            TP_100, FP_100, N_100 =  sceltaN(list(mse), y_test_step_i, week, threshold_fixed, 100,lineages_know[ind_prc])
            fractions_100.append([TP_100, FP_100, N_100])
            # Graphs
            graphic_fraction(fractions_100, 100, path_salvataggio_file)

            # The k-mers importance
            i_anomaly = np.where(y_test_i_predict == -1)[0]
            features_no_zero = [features[i] for i in i_no_zero]
            selection_kmers(test_x_predictions, test_step_i, features_no_zero, y_test_i_predict, id_identifier,'Summary_'+str(starting_week+week)+'.csv')

            #  selection the anomaly defined by the model
            mse_top100_anomaly = mse[i_anomaly]
            lineage_top100_anomaly = lineages_error_test[i_anomaly]

            # Select the top 100 and sort the mse
            size = 100
            if len(i_anomaly) < 100:
                size = len(i_anomaly)
            top_indices_100 = mse_top100_anomaly.argsort()[-size:][::-1]
            lineages_predetti_top_100 = lineage_top100_anomaly[top_indices_100]

            # Filtering about the knowledge
            prediction = list(-np.ones(size))
            prediction_filtering = lookup_post(prediction, lineages_predetti_top_100, lineages_know[ind_prc])

            # Find the true anomalies after filtering
            prediction_filtering = np.array(prediction_filtering)
            index_anomaly_filter = np.where(prediction_filtering == -1)[0]
            lineages_predetti_top_100 = lineages_predetti_top_100[index_anomaly_filter]
            lineages_counter_top_100 = Counter(lineages_predetti_top_100)
            total_100 = sum(lineages_counter_top_100.values())
            lineage_percentage_100 = {k: (v / total_100) * 100 for k, v in lineages_counter_top_100.items()}

            # list prediction
            summary_100_anomalies.append([week,lineages_counter_top_100])
            summary_100_anomalies_percentage.append([week,lineage_percentage_100])

            # Write the file in txt the prediction
            with open(path_salvataggio_file+'/TOP_100_FILTERING.txt', 'w') as file:
                # Scrivi ogni elemento della lista in una nuova riga nel file
                for elemento in summary_100_anomalies:
                    file.write(elemento + '\n')

            # Write the file in txt the prediction precision
            with open(path_salvataggio_file + '/TOP_100_FILTERING_PERCENTAGE.txt', 'w') as file:
                # Scrivi ogni elemento della lista in una nuova riga nel file
                for elemento in summary_100_anomalies_percentage:
                    file.write(elemento + '\n')

            # COSTRUZIONE TRAINING
            train_step_completo=np.concatenate((train_step_completo, test_step_completo)) # solo 0 e 1
            #train_with_class_completo=np.concatenate((train_with_class_completo, test_with_class_completo))
            lineages_train=np.concatenate((lineages_train, lineages_test)) # solo stringhe
            y_test_dict_predictedclass[starting_week + week] = y_test_i_predict
            y_test_voc_predict = np.array(y_test_i_predict)[i_voc]

            logging.info("Number of lineage in week:" + str(test_step_i.shape[0]))
            print("Number of lineage in week:" + str(test_step_i.shape[0]))
            logging.info("Number of lineage of concern in week:" + str(len([x for x in y_test_fclass_i if x == -1])))
            print("Number of lineage of concern in week:" + str(len([x for x in y_test_fclass_i if x == -1])))
            logging.info("Distribution of lineage of concern:" + str(Counter(y_test_step_i)))
            print("Distribution of lineage of concern:" + str(Counter(y_test_step_i)))
            logging.info("Number of lineage predicted as anomalty:" + str(
            len([x for x in y_test_dict_predictedclass[starting_week + week] if x == -1])))
            print("Number of lineage predicted as anomalty:" + str(
                len([x for x in y_test_dict_predictedclass[starting_week + week] if x == -1])))
            acc_voc = len([x for x in y_test_voc_predict if x == -1])
            logging.info("Number of lineages of concern predicted as anomalty:" + str(acc_voc))
            print("Number of lineages of concern predicted as anomalty:" + str(acc_voc))

            for k in lineage_dict.keys():
                i_k = np.where(np.array(y_test_step_i) == k)[0]
                logging.info('Number of ' + k + ' lineage:' + str(len(i_k)) + '; predicted anomalty=' + str(
                    len([x for x in y_test_i_predict[i_k] if x == -1])))
                print('Number of ' + k + ' lineage:' + str(len(i_k)) + '; predicted anomalty=' + str(
                    len([x for x in y_test_i_predict[i_k] if x == -1])))
                h = len([x for x in y_test_i_predict[i_k] if x == -1])

        # saving results for this comb of param of the oneclass_svm
        results = {'y_test_variant_type': y_test_dict_variant_type,
               'y_test_final_class': y_test_dict_finalclass,
               'y_test_predicted_class': y_test_dict_predictedclass}
    results_fine_tune.append(results)

    print('---------------------------------Vector of predictions total----------------------------------------------------------------')
    print(summary_100_anomalies)
    print('---------------------------------Vector of predictions percentage----------------------------------------------------------------')
    print(summary_100_anomalies_percentage)
    print('---------------------------------Fractions top 100---------------------------------------------------------------------------')
    print(fractions_100)
    print('----------------------------------------------------------------------------------------------------------------------------')

    # Precision, Recall, info = calcola_prc(info_PCR, path_salvataggio_file)
    #
    # print(
    #     '---------------------------------Precision---------------------------------------------------------------------------')
    # print(Precision)
    # print(
    #     '----------------------------------------------------------------------------------------------------------------------------')
    #
    # print(
    #     '---------------------------------Recall---------------------------------------------------------------------------')
    # print(Recall)
    # print(
    #     '----------------------------------------------------------------------------------------------------------------------------')

    # THE BEST AND WORSE
    best_worst(path_salvataggio_file) #world


    # clalcolo grafici e matrice
    A = sensitivity(measure_sensibilit,
                    path_salvataggio_file)  # funzione per la creazione dei grafici e delle matrici di confusione
    print('Computed the confusion matrix')

    #calcolo distanza
    distanza=scoperta(measure_sensibilit) # calcolo di quanto prima becco le varianti del massimo lineage
    distanza_np=np.array(distanza)
    distanza_lista = list(distanza_np)
    with open(path_salvataggio_file + '/distnce_prediction.txt', 'w') as file:
        # Scrivi ogni elemento della lista in una nuova riga nel file
        for elemento in distanza_lista:
            file.write(elemento + '\n')
    print('-----------------------------------------Prediction weeks before threshold-------------------------------------------------------')
    print(distanza_np)
    print('---------------------------------------------------------------------------------------------------------------------------')

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-p", "--pathdrive", dest="path_drive",

                      help="path to drive example: path/drive/", default="/blue/salemi/share/varcovid/dataset_febb_2023_little/")   # default
    parser.add_option("-c", "--csv", dest="csv_path",

                      help="path to CSV file metadata", default="/blue/salemi/share/varcovid/filtered_metadatataset_010223_edit_200323.csv")

    parser.add_option("-k","--kmers",dest="kmers",
                      help="path of file kmers",default='/blue/salemi/share/varcovid/dataset_febb_2023_little/1/EPI_ISL_14307752.csv')

    parser.add_option("-s", "--pathsave ", dest="path_save",
                      help="path where we can save the file", default='/blue/salemi/share/varcovid/PAPAER_GROSSO/RISULTATI/WORLD_25_TH')

    parser.add_option("-m", "--mantain ", dest="rate_mantain",
                      help="rate for mantain the k-mers", default=0.05)

    parser.add_option("-e", "--Epoch ", dest="number_epoch",
                      help="number of epochs", default=10)

    parser.add_option("-b", "--Batchsize ", dest="batch_size",
                      help="number of batchsize in the first week", default=256)

    parser.add_option("-d", "--encoding dimension ", dest="encoding_dim",
                      help="encodin dimention", default=1024)

    parser.add_option("-r", "--reduction facor ", dest="red_factor",
                      help="red_factor", default=1e-7)

    (options, args) = parser.parse_args()
    main(options)
