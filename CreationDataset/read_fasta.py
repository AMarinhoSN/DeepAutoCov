def read_fasta(file):
    sequences = []
    with open(file, 'r') as f:
        current_sequence = ''
        started = False
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if started:
                    sequences.append(current_sequence)
                started = True
                current_sequence = ''
            else:
                current_sequence += line
        if current_sequence:
            sequences.append(current_sequence)
    return sequences