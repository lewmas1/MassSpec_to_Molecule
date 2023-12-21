import re
import os
import numpy as np
from sklearn.model_selection import train_test_split
from selfies import encoder, decoder
import tqdm


def parse_msp(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        entries = file.read().split('\n\n')  # Each entry is separated by a blank line

    for entry in entries:
        lines = entry.strip().split('\n')
        record = {}
        peaks = []
        try:
            record = {line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip() for line in lines if ':' in line}
            peaks = [(float(m), float(i)) for line in lines if re.match(r'^\d', line) for m, i in [line.split()]]
        except Exception as e:
            print(f"Error parsing entry: {entry}. Skipping...")
            continue

        if 'SMILES' in record and peaks and 'MW' in record and float(record['MW']) <= 450:
            record['Peaks'] = peaks
            yield record


def spectra_process(original_list):
    new_list = np.zeros(500, dtype=int)
    for pair in original_list:
        index = int(round(pair[0]))
        if index < 500:
            new_list[index] = int(round(pair[1] / 10))
    new_list_str = ' '.join(str(num) for num in new_list)
    return new_list_str

def convert_smiles_to_selfies(smiles):
    selfies_string = encoder(smiles)
    parts = selfies_string[1:-1].split('][')
    selfies_string_with_space = ' '.join(parts)
    return selfies_string_with_space

def convert_selfies_to_smiles(selfies_string_with_space):
    selfies_parts = selfies_string_with_space.split(' ')
    selfies_string = '[' + ']['.join(selfies_parts) + ']'
    smiles = decoder(selfies_string)
    return smiles

def prep_data(spectra, tgt):
    data = []
    for i, spectrum_orig in enumerate(tqdm.tqdm(spectra)):
        tgt_temp = tgt[i]
        spectrum_array = np.array(list(map(int, spectrum_orig.split())))
        data.append([tgt_temp, " ".join(spectrum_array.astype(str))])
    return np.array(data)

def split_train_test_val(data):
    train_set, test_set = train_test_split(data, test_size=0.2)
    train_set, val_set = train_test_split(train_set, test_size=0.1)
    return train_set, test_set, val_set

def save_data(train_data, test_data, val_data, path):
    np.savetxt(os.path.join(path, "src-train.txt"), train_data[:, 1], fmt="%s")
    np.savetxt(os.path.join(path, "tgt-train.txt"), train_data[:, 0], fmt="%s")
    np.savetxt(os.path.join(path, "src-test.txt"), test_data[:, 1], fmt="%s")
    np.savetxt(os.path.join(path, "tgt-test.txt"), test_data[:, 0], fmt="%s")
    np.savetxt(os.path.join(path, "src-val.txt"), val_data[:, 1], fmt="%s")
    np.savetxt(os.path.join(path, "tgt-val.txt"), val_data[:, 0], fmt="%s")

msp_files = ['data/MassBank_NIST.msp', 'data/MassBank_RIKEN.msp']
msp_data = [parse_msp(file) for file in msp_files]

data = []
for msp in msp_data:
    for line in msp:
        try:
            if line['SMILES'] != 'N/A':
                spectra = spectra_process(line['Peaks'])
                smiles = convert_smiles_to_selfies(line['SMILES'])
                data.append((spectra, smiles))
        except Exception as e:
            print(f"Error processing entry: {line['SMILES']}. Skipping...")
            continue

train_set, test_set, val_set = split_train_test_val(prep_data([item[0] for item in data], [item[1] for item in data]))
save_data(train_set, test_set, val_set, "data")