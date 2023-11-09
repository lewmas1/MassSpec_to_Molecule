# MassSpec to Molecule

A transformer based neural network to train and predict molecular structures (SMILEs) from mass spectra data. Setup to be trained on data from [MassBank](https://github.com/MassBank/MassBank-data/releases). 

Adapted code from similar project to predict molecular structure (SMILEs) from IR data [here](https://github.com/rxn4chemistry/rxn-ir-to-structure).

Install dependencies:
```
pip install -r requirements.txt
```

## Training the model

This section explains how to train the model.

### Download data

First download the latest MassBank db from [here](https://github.com/MassBank/MassBank-data/releases) and place in the `data` folder

### Processing the data

To process the data run the `data_processing.py` script to create the train, test and evaluation datasets saved as .txt

### Training

Train the model by running the `training.py` script which requires both the data files created by the `data_processing.py` and an openNMT template. Two templates have been provided `templates/transformer_template.yaml` which is a larger model to be trained on a GPU and `templates/transformer_template_cpu.yaml` which is a smaller model for training on a CPU.

Once training is complete it should save the trained models in the `model` directory.

## Scoring and inference

TBA