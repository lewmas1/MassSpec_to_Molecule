import re
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
import os

def parse_msp(file_path):
    with open(file_path, 'r') as file:
        entries = file.read().split('\n\n')  # Each entry is separated by a blank line

    data = []
    for entry in entries:
        lines = entry.strip().split('\n')
        record = {}
        peaks = []
        for line in lines:
            if line.startswith('MW:'):
                record['MW'] = line.split(':', 1)[1].strip()
            elif line.startswith('Name:'):
                record['Name'] = line.split(':', 1)[1].strip()
            elif line.startswith('SMILES:'):
                record['SMILES'] = line.split(':', 1)[1].strip()
            elif line.startswith('Num Peaks:'):
                record['Num_Peaks'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('MW:'):
                mass.append(float(line.split(':', 1)[1].strip()))
            elif re.match(r'^\d', line):  # Starts with a number, it's a peak line
                m, i = line.split()
                peaks.append((float(m), float(i)))

        if record and 'SMILES' in record and peaks and 'MW' in record and float(record['MW']) <= 450:
            record['Peaks'] = peaks
            data.append(record)

    return data
def spectra_process(original_list):

    new_list = [0] * 500
    for pair in original_list:
        index = int(round(pair[0]))
        if index < 500:
            new_list[index] = int(round(pair[1]/10))
    new_list_str = ' '.join(str(num) for num in new_list)
    return new_list_str

def split_smiles(smile: str) -> str:
    pattern_full = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    regex = re.compile(pattern_full)
    tokens = [token for token in regex.findall(smile)]

    if smile != "".join(tokens):
        raise ValueError(
            "Tokenised smiles does not match original: {} {}".format(tokens, smile)
        )

    return " ".join(tokens)


def prep_data(spectra, tgt):
    data = []

    for i, spectrum_orig in enumerate(tqdm.tqdm(spectra)):
        tgt_temp = tgt[i]

        # Convert spectrum_orig from a string to a NumPy array
        spectrum_array = np.array(list(map(int, spectrum_orig.split())))

        data.append([tgt_temp, " ".join(spectrum_array.astype(str))])

    return np.array(data)


def split_train_test_val(data):
    train_set, test_set = train_test_split(data, test_size=0.2)
    train_set, val_set = train_test_split(train_set, test_size=0.1)

    return train_set, test_set, val_set


def save_data(train_data, test_data, val_data, path):
    with open(os.path.join(path, "src-train.txt"), "w") as f:
        for item in train_data[:, 1]:
            f.write(f"{item}\n")

    with open(os.path.join(path, "tgt-train.txt"), "w") as f:
        for item in train_data[:, 0]:
            f.write(f"{item}\n")

    with open(os.path.join(path, "src-test.txt"), "w") as f:
        for item in test_data[:, 1]:
            f.write(f"{item}\n")

    with open(os.path.join(path, "tgt-test.txt"), "w") as f:
        for item in test_data[:, 0]:
            f.write(f"{item}\n")

    with open(os.path.join(path, "src-val.txt"), "w") as f:
        for item in val_data[:, 1]:
            f.write(f"{item}\n")

    with open(os.path.join(path, "tgt-val.txt"), "w") as f:
        for item in val_data[:, 0]:
            f.write(f"{item}\n")


msp_files = ['data/MassBank_NIST.msp', 'data/MassBank_RIKEN.msp']
msp_data = [parse_msp(file) for file in msp_files]

data = [(spectra_process(line['Peaks']), split_smiles(line['SMILES'])) for msp in msp_data for line in msp]

train_set, test_set, val_set = split_train_test_val(prep_data([item[0] for item in data], [item[1] for item in data]))

save_data(train_set, test_set, val_set, "data")