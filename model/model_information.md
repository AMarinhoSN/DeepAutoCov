In this file you can find some information about the code. 
The main file is "Main_prediction_AE" where the simulation is implemented and the parameters of model identified. Several functions are called in the main. In particular: 
1) model_dl.py: definition of model, layers, decision function;
2) Autoencoder_training.py: training of the model;
3) Best_worse.py: definition of the lineages that the model predict in advance
4) PRC_Graphic_curve.py: function to graph a PRC curve
5) PRC_curve.py: function to calculate the PRC_curve
6) fraction_mail.py: function to calculate the precision in the top100 during the week
7) falsepositive.py: function to calculate the falsepositive rate and others parameters
8) barplot_laboratory.py: barplot to calculate precision on top100
9) filter_dataset.py: function to obtain new dataset to retrain the model during the retraining week
10) weeks_retraining.py: we definne the week of retraining
11) scoperta.py: calculte weeks in advance of prediction about the lineages
12) test_normality_error: calculate the treshold and if errors are distribuated like a Gaussian or not 
