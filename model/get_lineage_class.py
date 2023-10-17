
def get_lineage_class(metadata, id_list):
    variant_name_list = []
    for id in id_list:
        variant_name_list.append(metadata[metadata['Accession.ID'] == id]['Pango.lineage'].values[0])
    return variant_name_list
