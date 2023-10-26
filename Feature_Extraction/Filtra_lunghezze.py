def filtra_sequenze(sequenze, lunghezza_minima, lunghezza_massima): # We ecide the maximum and minimum length i.e. I put the general median - 20 amino acids and + 20 amino acids 
    indici = [i for i, seq in enumerate(sequenze) if lunghezza_minima <= len(seq) <= lunghezza_massima]
    sequenze_valide = [seq for i, seq in enumerate(sequenze) if lunghezza_minima <= len(seq) <= lunghezza_massima]
    return indici, sequenze_valide

# sequenza=['ASTREFGIHILMONOPRST','ASTREFGIHILMONOPRST','A','BVG','ASTREFGIHILMONOPRST']
# indici,sequenze_valide=filtra_sequenze(sequenza, 4, 22)
# print(indici)
# print(sequenze_valide)
