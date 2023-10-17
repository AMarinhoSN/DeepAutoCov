def filtra_sequenze(sequenze, lunghezza_minima, lunghezza_massima): # La lunghezza massima e minima la ecidiamo noi cioè io metto la mediana generale - 20 amminoacidi e + 20 amminoacidi
    indici = [i for i, seq in enumerate(sequenze) if lunghezza_minima <= len(seq) <= lunghezza_massima]
    sequenze_valide = [seq for i, seq in enumerate(sequenze) if lunghezza_minima <= len(seq) <= lunghezza_massima]
    return indici, sequenze_valide

# sequenza=['ASTREFGIHILMONOPRST','ASTREFGIHILMONOPRST','A','BVG','ASTREFGIHILMONOPRST']
# indici,sequenze_valide=filtra_sequenze(sequenza, 4, 22)
# print(indici)
# print(sequenze_valide)