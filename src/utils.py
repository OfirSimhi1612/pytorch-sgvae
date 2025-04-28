import csv
import pickle
import random

import h5py
import numpy as np
from rdkit import Chem, RDLogger
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

from src.config.hyper_parameters import hyper_params

RDLogger.DisableLog("rdApp.*")


def write_csv(d, path):
    """
    Writes the results to the dict d
    """
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(d.keys())
        writer.writerows(zip(*d.values()))


class LoadSmilesData:

    def __init__(self, labels_path, normalization):

        random.seed(42)  # To ensure the same labels are picked everytime
        with open(labels_path, "rb") as data:
            f = pickle.load(data)

        smiles = list(f.loc[:, "SMILES_GDB-17"])
        prop = list(f.loc[:, hyper_params["prop"]])
        n_atoms = list(f.loc[:, "number_of_atoms"])

        # choosing random ids for train and test
        ids = list(range(len(smiles)))
        random.shuffle(ids)
        chunk = int(0.04 * len(smiles))

        self._ids_train = sorted(ids[chunk:])
        self._ids_test = sorted(ids[0:chunk])

        self._smiles_train = [smiles[i] for i in self._ids_train]
        self._smiles_test = [smiles[i] for i in self._ids_test]
        property_train = np.asarray([prop[i] for i in self._ids_train]).reshape(-1, 1)
        property_test = np.asarray([prop[i] for i in self._ids_test]).reshape(-1, 1)

        # Divinding the property values by the number of atoms in each molecule
        n_atoms_train = np.asarray([n_atoms[i] for i in self._ids_train]).reshape(-1, 1)
        property_train_div = property_train / n_atoms_train
        n_atoms_test = np.asarray([n_atoms[i] for i in self._ids_test]).reshape(-1, 1)
        property_test_div = property_test / n_atoms_test

        # normalizing the property values
        if normalization.lower().strip() == "minmax":
            minmaxscaler = MinMaxScaler()
            self._property_train_normalized = minmaxscaler.fit_transform(property_train)
            self._property_test_normalized = minmaxscaler.transform(property_test)
            self._property_train_normalized_div = minmaxscaler.fit_transform(
                property_train_div
            )
            self._property_test_normalized_div = minmaxscaler.transform(
                property_test_div
            )

        elif normalization.lower().strip() == "standard":
            standardscaler = StandardScaler()
            self._property_train_normalized = standardscaler.fit_transform(
                property_train
            )
            self._property_test_normalized = standardscaler.transform(property_test)
            self._property_train_normalized_div = standardscaler.fit_transform(
                property_train_div
            )
            self._property_test_normalized_div = standardscaler.transform(
                property_test_div
            )

        elif normalization.lower().strip() == "none":
            self._property_train_normalized = property_train
            self._property_test_normalized = property_test
            self._property_train_normalized_div = property_train_div
            self._property_test_normalized_div = property_test_div

        else:
            raise ValueError(
                "Invalid normalization method. Current options are minmax, standard and none"
            )

    def smiles_train(self):
        return self._smiles_train

    def smiles_test(self):
        return self._smiles_test

    def property_train(self):
        return self._property_train_normalized.flatten().tolist()

    def property_test(self):
        return self._property_test_normalized.flatten().tolist()

    def property_train_div(self):
        return self._property_train_normalized_div.flatten().tolist()

    def property_test_div(self):
        return self._property_test_normalized_div.flatten().tolist()

    def ids_train(self):
        return self._ids_train

    def ids_test(self):
        return self._ids_test


class AnnealKL:
    """
    Anneals weight to the kl term in a cyclical manner
    """

    def __init__(self, n_epoch, start=0, stop=1, n_cycle=2, ratio=0.5):
        self.L = np.ones(n_epoch)
        period = n_epoch / n_cycle
        step = (stop - start) / (period * ratio)

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop:
                self.L[int(i + c * period)] = 1.0 / (1.0 + np.exp(-(v * 12.0 - 6.0)))
                v += step
                i += 1

    def beta(self, epoch):
        return self.L[epoch]


class TrainQM9dataset(Dataset):
    """
    Loads the QM9 training data set for the dataloader
    """

    def __init__(self, dataset_path, labels_path, normalization, div=True):
        train_data = []
        h5f = h5py.File(dataset_path, "r")
        data = h5f["data"][:].astype(np.float32)
        h5f.close()

        train_labels = LoadSmilesData(labels_path, normalization)
        if hyper_params["prop_div"]:
            labels = train_labels.property_train_div()
        else:
            labels = train_labels.property_train()

        for i in range(len(data)):
            train_data.append([data[i], labels[i]])
        self.data = train_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
