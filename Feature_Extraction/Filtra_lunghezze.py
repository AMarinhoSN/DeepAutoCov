
# filtra_sequenze: function to filter the length of sequences 
# INPUT:
#    1) sequences: list
#    2) length_minimum : minimum length acceptable 
#    3) length_maximum : maximum length acceptable 
def filtra_sequenze(sequences, length_minimum, length_maximum): # We ecide the maximum and minimum length i.e. I put the general median - 20 amino acids and + 20 amino acids 
    index = [i for i, seq in enumerate(sequences) if length_minimum <= len(seq) <= length_maximum]
    sequences_valid = [seq for i, seq in enumerate(sequences) if length_minimum <= len(seq) <= length_maximum]
    return index, sequences_valid

# sequenza=['ASTREFGIHILMONOPRST','ASTREFGIHILMONOPRST','A','BVG','ASTREFGIHILMONOPRST']
# indici,sequenze_valide=filtra_sequenze(sequenza, 4, 22)
# print(indici)
# print(sequenze_valide)
