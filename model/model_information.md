In this file you can find some information about the codes. 
In the script <code>Main_prediction_AE</code> the simulation is implemented and the parameters of Autoencoder identified. Several functions are called in the main. In particular: 
1) <code>model_dl.py</code>: definition of model, layers, decision function;
2) <code>Autoencoder_training.py</code>: training of the model;
3) <code>Best_worse.py</code>: definition of the lineages that the model predict in advance;
4) <code>PRC_Graphic_curve.py</code>: function to graph a PRC curve;
5) <code>PRC_curve.py</code>: function to calculate the PRC_curve;
6) <code>fraction_mail.py</code>: function to calculate the precision in the top100 during the week;
7) <code>falsepositive.py</code>: function to calculate the falsepositive rate and others parameters;
8) <code>barplot_laboratory.py</code>: barplot to calculate precision on top100;
9) <code>filter_dataset.py</code>: function to obtain new dataset to retrain the model during the retraining week;
10) <code>weeks_retraining.py</code>: we definne the week of retraining;
11) <code>scoperta.py</code>: calculte weeks in advance of prediction about the lineages;
12) <code>test_normality_error.py</code>: calculate the treshold and if errors are distribuated like a Gaussian or not;
13) <code>kmers_errors.py</code>:Identify erroneous predicted k-mers.  
