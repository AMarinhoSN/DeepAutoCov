# function that return a lineage name 
# INPUT 
# metadata: file csv that contains the information of sequences
# id_list: list that contains the id of sequences
# OUTPUT
# variant_name_list: list that contains the name of lineages 
def get_lineage_class(metadata, id_list):
    variant_name_list = []
    for id in id_list:
        variant_name_list.append(metadata[metadata['Accession.ID'] == id]['Pango.lineage'].values[0])
    return variant_name_list
