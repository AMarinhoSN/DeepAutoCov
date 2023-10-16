# funxione che mi mappa i lineage
def map_lineage_to_finalclass(class_list, non_neutral):
    # -1 -> non-neutral
    # 1 -> neutral
    final_class_list = []
    for c in class_list:
        if c in non_neutral:
            final_class_list.append(-1)
        else:
            final_class_list.append(1)
    return final_class_list