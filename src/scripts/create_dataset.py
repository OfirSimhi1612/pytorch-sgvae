"""
This script processes SMILES strings from the QM9 dataset and converts them into one-hot encoded representations
based on a predefined grammar. The resulting one-hot encoded dataset is saved as an HDF5 file for later use.

Sections:
1. Load and split the QM9 dataset into training and testing sets.
2. Define helper functions for converting SMILES strings to one-hot encoded representations.
3. Main function to process the training SMILES strings and save the one-hot encoded dataset.
"""

import pickle
import random

import h5py
import nltk
import numpy as np
from tqdm import tqdm

import src.grammar_preprocessor as grammar_preprocessor
from src import grammar
from src.config.hyper_parameters import hyper_params

# Set random seed for reproducibility
random.seed(42)

# Load the QM9 dataset
with open("data/QM9_STAR.pkl", "rb") as data:
    dataset = pickle.load(data)

# Extract SMILES strings from the dataset
smiles_list = list(dataset.loc[:, "SMILES_GDB-17"])

# Split dataset into training and testing sets
ids = list(range(len(smiles_list)))
random.shuffle(ids)

test_chunk = int(hyper_params["testing_split"] * len(smiles_list))
train_ids = sorted(ids[test_chunk:])
test_ids = sorted(ids[:test_chunk])

training_smiles = [smiles_list[i] for i in train_ids]  # SMILES for training
testing_smiles = [smiles_list[i] for i in test_ids]  # SMILES for testing

# Define constants
input_dim = hyper_params["input_dim"]  # Maximum sequence length for one-hot encoding
num_grammar_rules = len(
    grammar.GCFG.productions()
)  # Number of grammar production rules

# Map grammar productions to indices
production_to_index = {prod: ix for ix, prod in enumerate(grammar.GCFG.productions())}
parser = nltk.ChartParser(grammar.GCFG)

# Tokenize SMILES strings
tokenizer = grammar_preprocessor.get_zinc_tokenizer(grammar.GCFG)


def to_one_hot(smiles):
    """
    Convert a list of SMILES strings into one-hot encoded representations based on the grammar.

    Args:
        smiles (list): List of SMILES strings.

    Returns:
        np.ndarray: One-hot encoded representation of the SMILES strings.
    """
    assert isinstance(smiles, list), "Input must be a list of SMILES strings."

    tokens = map(tokenizer, smiles)

    # Parse tokens into production sequences using the grammar parser
    # Each tokenized SMILES string is parsed into a parse tree, which represents the hierarchical structure of the string based on the grammar rules.
    # The parse tree is then converted into a sequence of production rules (grammar rules) that were applied to generate the tree.
    parse_trees = [parser.parse(t).__next__() for t in tokens]
    production_sequences = [tree.productions() for tree in parse_trees]

    # Convert production sequences to indices
    indices = [
        np.array([production_to_index[prod] for prod in sequence], dtype=int)
        for sequence in production_sequences
    ]

    # Initialize one-hot encoded array as a 3D tensor with dimensions:
    # (number of SMILES, input dimension, number of grammar production rules)
    one_hot = np.zeros((len(indices), input_dim, num_grammar_rules), dtype=np.float32)

    # Populate the one-hot encoded array for each SMILES string
    # For each SMILES string, the sequence of production rule indices is used to set the corresponding positions in the one-hot array to 1.0.
    # The first dimension (i) corresponds to the SMILES string index in the batch.
    # The second dimension (np.arange(num_productions)) corresponds to the position in the sequence.
    # The third dimension (sequence_indices) corresponds to the grammar production rule index.
    # Padding is applied to the remaining positions in the sequence (from num_productions to input_dim) by setting the last index (-1) to 1.0.
    for i, sequence_indices in enumerate(indices):
        num_productions = len(sequence_indices)
        one_hot[i, np.arange(num_productions), sequence_indices] = 1.0
        one_hot[i, np.arange(num_productions, input_dim), -1] = 1.0  # Padding

    return one_hot


def main():
    """
    Main function to process the training SMILES strings and save the one-hot encoded dataset in batches.
    """
    # Open the HDF5 file and create the dataset
    with h5py.File("data/qm9_grammar_dataset.h5", "w") as h5f:
        dataset = h5f.create_dataset(
            "data",
            shape=(len(training_smiles), input_dim, num_grammar_rules),
            dtype=np.float32,
        )

        # Process SMILES strings in batches with a progress bar
        with tqdm(
            total=len(training_smiles), desc="Processing SMILES", unit=" smiles"
        ) as pbar:
            for i in range(0, len(training_smiles), 100):
                one_hot_batch = to_one_hot(training_smiles[i : i + 100])
                dataset[i : i + 100, :, :] = one_hot_batch  # Write directly to the file
                pbar.update(100)

    print("One-hot encoded dataset saved to data/qm9_grammar_dataset.h5")


if __name__ == "__main__":
    main()
