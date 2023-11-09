#!/usr/bin/env python3
import argparse
import deepautocov
import os
parser = argparse.ArgumentParser(
        description="Feature extraction script"
    )

requiredNamed = parser.add_argument_group("required named arguments")
requiredNamed.add_argument("-f", "--fasta", dest="fasta_path",
    help="path to a SPIKE FASTA file")

requiredNamed.add_argument("-c", "--csv", dest="csv_path",
    help="path to GISAID metadata CSV file")

parser.add_argument("-n","--continent",dest="continent_list",
    help="list of continents of interest [DEFAULT=['/']]", default=['/'])

parser.add_argument("-m", "--minlen ", dest="min_length",
    help="minimum length of sequence", default=1000)

parser.add_argument("-l", "--median_limit ", dest="med_limit",
    help="median range", default=30)

parser.add_argument("-o", "--output_dir", dest="save_path",
    help="set ouput dir [DEFAULT=<current working directory>]", default=os.getcwd())

args = parser.parse_args()


deepautocov.data_filtration_kmers_rewrite.main(args)