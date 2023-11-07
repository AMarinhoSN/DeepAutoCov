import numpy as np
import csv


def write_csv_dataset(array,l):
    # Definition of column names as a list of strings.
    nomi_colonne = ['Virus.name','Not.Impo','format','Type','Accession.ID','Collection.date','Location','Additional.location.information','Sequence.length','Host','Patient.age','Gender','Clade','Pango.lineage','Pangolin.type','Variant','AA.Substitutions','Submission.date','Is.reference.','Is.complete.','Is.high.coverage.','Is.low.coverage.','N.Content']
    # Opening the CSV file in write mode and defining the writer.
    with open('filtered_metadatataset_'+l+'.csv', "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")

        # Writing header row with column names.
        writer.writerow(nomi_colonne)

        # Writing data rows
        for riga in array:
            writer.writerow(riga)
