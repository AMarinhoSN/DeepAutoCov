from load_data import *
from get_lineage_class import *
from map_lineage_to_finalclass import *
from optparse import OptionParser
from Calcolo_week import *
from lineages_validi import *
from weeks_retraining import *
from datetime import *
from Autoencoder_training_GPU import *
import logging
from sklearn.model_selection import ParameterGrid
from sensitivity import *
from falsepositive import *
from fraction_mail import *
from barplot_laboratory import *
from scoperta import *
from filter_dataset import *
from test_normality_error import *
import gc
from PRC_curve import *
from model_dl import *
from Autoencoder_training import *
from PRC_Graphic_curve import *
from Best_worse import *
from plot_smooth import *
from kmers_error import *


def main(options):
    # memory control
    gc.enable()
    # use GPU
    strategy = tf.distribute.MirroredStrategy()

    # define a list that we will use during the code
    beta = []  # for 100
    measure_sensibilit = []  
    NF = []  # Number of features
    results_fine_tune = []
    fractions_100 = []
    mse_prc_curve = []  
    true_class_prc_curve = []  
    info_PCR = []  
    ind_prc = 0

    
    dir_week =str(options.path_drive) #path Dataset 
    
    metadata = pd.read_csv(str(options.csv_path)) # read the file 

    # columns in metadata file
    col_class_lineage = 'Pango.lineage'
    col_submission_date = 'Collection.date'
    col_lineage_id = 'Accession.ID'

    
    valid_lineage,valid_lineage_prc,dizionario_lineage_settimane=lineages_validi() # return lineages of interest

    metadata[col_class_lineage] = metadata[col_class_lineage].apply(lambda x: 'unknown' if x not in valid_lineage else x) 
    id_unknown = metadata[metadata[col_class_lineage] == 'unknown'][col_lineage_id].tolist() 

    # week of retraining
    retraining_week, retraining_week_false_positive=weeks_retrain()

    # K-mers
    header = pd.read_csv(str(options.kmers), nrows=1) 
    features = header.columns[1:].tolist()  # k-mers
    print('-----------------------------------------------------------------------------------')
    print('k-mers : ' + str(len(features)))
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
    starting_week = 1 

    # Loading first training step
    df_trainstep_1, train_w_list = load_data(dir_week, [starting_week])
    train_step1 = df_trainstep_1.iloc[:, 1:len(df_trainstep_1.columns)].to_numpy()

    # define the features mantain
    sum_train = np.sum(train_step1, axis=0)
    keepFeature=sum_train/len(train_step1)
    i_no_zero = np.where(keepFeature >= options.rate_mantain)[0] # tengo le feature che sono diverse di alamneo il N%

    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('features defined :' + str((len(i_no_zero))))
    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')

    y_train_initial = metadata[metadata[col_lineage_id].isin(df_trainstep_1.iloc[:, 0].tolist())][col_class_lineage]
    y_train_class = map_lineage_to_finalclass(y_train_initial.tolist(), lineage_of_interest)
    counter_i = Counter(y_train_initial)  # at the beginning, all the lineages were "unknown"=neutral

    # filtering out features with all zero
    train_step_completo = train_step1
    train = train_step1[:, i_no_zero] # Per il primo allenamento metto il train
    lineages_train=np.array(y_train_initial.tolist()) # lineages

    tf.random.set_seed(10)
    # Creation the Autoencoder models

    # parameters of autoencoder
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
        print('Trained the model : ')
        print(history)
        # CALCOLO P-VALUE
        info, mse_tr = test_normality(autoencoder, train) # we calultae the mse in a normal dataset
        for week in range(1, 159):  # metadata['week'].max()
            if week in retraining_week:
                logging.info('----> RETRAINING <-----')

                ind_prc = ind_prc + 1
                soglia = threshold_fixed
                min_prob = min(mse_prc_curve)
                max_prob = max(mse_prc_curve)
                print('--------PRC--------')
                info_pcr = evaluate_pcr(true_class_prc_curve, mse_prc_curve, soglia, min_prob, max_prob)
                info_PCR.append(info_pcr)

                # We create a new training set for retrain the network
                train_model_value = train_step_completo # soloziono solo i kmaers
                classi=lineages_train #seleziono solo i valori
                sum_train = np.sum(train_model_value, axis=0)
                keepFeature = sum_train / len(train_model_value)
                i_no_zero = np.where(keepFeature > options.rate_mantain)[0]

                number_feature = len(i_no_zero)
                print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
                print('features defined :' + str((len(i_no_zero))))
                print('---------------------------------------------------------------------------------------------------------------------------------------------------------')

                train_model_value = train_model_value[:, i_no_zero]

               
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

                print('Trained the neural network : ')
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
                0]  

            # selection the first 100 with highest mse
            TP_100, FP_100, N_100 = sceltaN(list(mse), y_test_step_i, week, threshold_fixed, 100)
            fractions_100.append([TP_100, FP_100, N_100])
            graphic_fraction(fractions_100, 100, path_salvataggio_file)

            # The k-mers importance
            features_no_zero = [features[i] for i in i_no_zero]
            selection_kmers(test_x_predictions, test_step_i, features_no_zero, y_test_i_predict, id_identifier,'Summary_'+str(starting_week+week)+'.csv')

            # COSTRUZIONE TRAINING
            train_step_completo=np.concatenate((train_step_completo, test_step_completo)) 
            #train_with_class_completo=np.concatenate((train_with_class_completo, test_with_class_completo))
            lineages_train=np.concatenate((lineages_train, lineages_test)) 
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
                Prova = [k, h, week]
                beta.append(Prova)
                measure_variant = [k, len(i_k), h, week]  
                measure_sensibilit.append(measure_variant)
                falsepositive(measure_sensibilit, retraining_week_false_positive, path_salvataggio_file)

        # saving results for this comb of param of the oneclass_svm
        results = {'y_test_variant_type': y_test_dict_variant_type,
               'y_test_final_class': y_test_dict_finalclass,
               'y_test_predicted_class': y_test_dict_predictedclass}
    results_fine_tune.append(results)

    print('---------------------------------Vector of information-------------------------------------')
    print(beta)
    print(measure_sensibilit)
    print('---------------------------------Fractions top 100---------------------------------------------------------------------------')
    print(fractions_100)
    print('----------------------------------------------------------------------------------------------------------------------------')

    Precision,Recall,info= calcola_prc(info_PCR,path_salvataggio_file)

    print('---------------------------------Precision---------------------------------------------------------------------------')
    print(Precision)
    print('----------------------------------------------------------------------------------------------------------------------------')

    print('---------------------------------Recall---------------------------------------------------------------------------')
    print(Recall)
    print('----------------------------------------------------------------------------------------------------------------------------')

    # THE BEST AND WORSE
    best_worst(path_salvataggio_file)

    # FALSE POSITIVE RATE  + boxplot
    FP_RATE_FINAL, Settimane_finali, TN_FINAL, TP_FINAL, FP_FINAL, FN_FINAL = falsepositive(measure_sensibilit,
                                                                                            retraining_week,
                                                                                            path_salvataggio_file)
    plot_sma(FP_RATE_FINAL, 4, path_salvataggio_file)



    # Others Graphs
    y_true_model0 = results_fine_tune[0]['y_test_final_class']
    y_predict_model0 = results_fine_tune[0]['y_test_predicted_class']

    fp_list = []
    n_list = []
    fn_list = []
    n_outlier_list = []

    for k in y_true_model0.keys():
        yt = np.array(y_true_model0[k])
        yp = np.array(y_predict_model0[k])

        i_inlier = np.where(yt == 1)[0]
        n_fp = len(np.where(yp[i_inlier] == -1)[0])

        fp_list.append(n_fp)
        n_list.append(len(i_inlier))

        i_outlier = np.where(yt == -1)[0]
        n_fn = len(np.where(yp[i_outlier] == 1)[0])
        fn_list.append(n_fn)
        n_outlier_list.append(len(i_outlier))

    tn_list = []
    tp_list = []

    prec_list = []
    recall_list = []
    spec_list = []
    f1_list = []
    for i in range(len(fp_list)):
        tp = n_outlier_list[i] - fn_list[i]
        tn = n_list[i] - fp_list[i]
        tn_list.append(tn)
        tp_list.append(tp)
        if tp + fp_list[i] != 0:
            prec = tp / (tp + fp_list[i])
        else:
            prec = 0

        if tp + fn_list[i] != 0:
            rec = tp / (tp + fn_list[i])
        else:
            rec = 0

        if tn + fp_list[i] != 0:
            spec = tn / (tn + fp_list[i])
        else:
            spec = 0

        if prec + rec != 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0
        f1_list.append(f1)
        spec_list.append(spec)
        prec_list.append(prec)
        recall_list.append(rec)

    df_conf = pd.DataFrame()
    df_conf['TN'] = tn_list
    df_conf['FP'] = fp_list
    df_conf['FN'] = fn_list
    df_conf['TP'] = tp_list
    df_conf['Precision'] = prec_list
    df_conf['Recall'] = recall_list
    df_conf['F1'] = f1_list
    df_conf['Specificity'] = spec_list

    # df_conf.to_csv('/mnt/resources/2022_04/2022_04/conf_mat_over_time.tsv', sep='\t', index=None)
    df_conf.to_csv(path_salvataggio_file+'/conf_mat_over_time.tsv', sep='\t', index=None)

    plt.figure(2)
    plt.bar(retraining_week, NF)
    plt.ylabel("No. of features")
    plt.xlabel("Retraining week")
    plt.title("No. of features during retraining")
    plt.savefig(path_salvataggio_file+'/number_of_features.png', dpi=350)

    x = np.arange(len(fp_list))
    fig, ax = plt.subplots(figsize=(32, 14))
    plt.rcParams.update({'font.size': 18})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    r = plt.bar(x, prec_list, width=0.35, alpha=0.8, color='#a32b15')
    # plt.bar_label(r, rotation=0, fontsize=16)
    plt.yticks(fontsize=25)
    plt.xticks(x, labels=[str(y + 2) for y in x], rotation=45, fontsize=20)
    plt.xlabel('Week', fontsize=25)

    plt.title('Precision in time  from 2019-12 to 2023-02', fontsize=25)
    plt.savefig(path_salvataggio_file+'/precision_in_time.png', dpi=350)

    fig, ax = plt.subplots(figsize=(32, 14))
    plt.rcParams.update({'font.size': 18})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    r = plt.bar(x, recall_list, width=0.35, alpha=0.8, color='#a32b15')
    # plt.bar_label(r, rotation=0, fontsize=16)
    plt.yticks(fontsize=25)
    plt.xticks(x, labels=[str(y + 2) for y in x], rotation=45, fontsize=20)
    plt.xlabel('Week', fontsize=25)

    plt.title('Recall in time  from 2019-12 to 2023-02', fontsize=25)
    plt.savefig(path_salvataggio_file+'/recall_in_time.png', dpi=350)

    fig, ax = plt.subplots(figsize=(32, 14))
    plt.rcParams.update({'font.size': 18})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    r = plt.bar(x, f1_list, width=0.35, alpha=0.8, color='#a32b15')
    # plt.bar_label(r, rotation=0, fontsize=16)
    plt.yticks(fontsize=25)
    plt.xticks(x, labels=[str(y + 2) for y in x], rotation=45, fontsize=20)
    plt.xlabel('Week', fontsize=25)

    plt.title('F1 in time  from 2019-12 to 2023-02', fontsize=25)
    plt.savefig(path_salvataggio_file+'/f1_in_time.png', dpi=350)

    x = np.arange(len(fp_list))

    fig, ax = plt.subplots(figsize=(32, 14))
    # plt.rcParams.update({'font.size':18})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    r = plt.bar(x, fp_list, width=0.35, alpha=0.8, color='#a32b15')
    #new_datalab = [str(x) for x in r.datavalues]
    #for i in range(len(new_datalab)):
        #if i % 2 != 0:
            #new_datalab[i] = ''
    # plt.bar_label(r, new_datalab, fontsize=25, padding=2)
    plt.yticks(fontsize=30)
    newlab_x = []
    for i in range(len(x)):
        if i % 5 == 0:
            newlab_x.append(str(x[i] + 2))
        else:
            newlab_x.append('')

    # plt.xticks(x, labels=[str(y+2) for y in x], rotation=45, fontsize=20)
    plt.xticks(x, labels=newlab_x, rotation=0, fontsize=28)
    plt.xlabel('Week', fontsize=25)
    plt.grid(axis='y')
    plt.title('Number of False Positive  from 2019-12 to 2023-02', fontsize=30)
    plt.tight_layout()
    plt.savefig(path_salvataggio_file+'/n_fp_v3.png', dpi=350)

    perc_fp = []
    for i in range(len(fp_list)):
        if n_list[i] != 0:
            perc_fp.append(round(fp_list[i] / n_list[i], 2))
        else:
            perc_fp.append(0)
    x = np.arange(len(fp_list))
    fig, ax = plt.subplots(figsize=(32, 14))
    plt.rcParams.update({'font.size': 18})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    r = plt.bar(x, perc_fp, width=0.35, alpha=0.8, color='#a32b15')
    #plt.bar_label(r, rotation=0, fontsize=16)
    plt.yticks(fontsize=25)
    plt.xticks(x, labels=[str(y + 2) for y in x], rotation=45, fontsize=20)
    plt.xlabel('Week', fontsize=25)

    plt.title('Percentage of False Positive from 2019-12 to 2023-02', fontsize=25)
    plt.savefig(path_salvataggio_file+'/perc_fp.png', dpi=350)

    # Calcolo flag
    prova = Calcolo_week(beta)  # Calcolo delle settimane per beccare 100 tipi di lineage
    type_of_lineage = list(prova[:, 0])
    Settimana = list([int(x) for x in prova[:, 1]])
    plt.figure(1)
    plt.bar(type_of_lineage, Settimana)
    plt.xlabel("Variant")
    plt.ylabel("No. of weeks")
    plt.title("Weeks for search 100 type of variant")
    plt.savefig(path_salvataggio_file + '/100_variant.png', dpi=350)

    # clalcolo grafici e matrice
    A = sensitivity(measure_sensibilit,
                    path_salvataggio_file)  # funzione per la creazione dei grafici e delle matrici di confusione
    print('ho calcolato le matrice di confusione')

    #calcolo distanza
    distanza=scoperta(measure_sensibilit) # calcolo di quanto prima becco le varianti del massimo lineage
    distanza_np=np.array(distanza)
    print('-----------------------------------------Predizione prima degli FDLs-------------------------------------------------------')
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
