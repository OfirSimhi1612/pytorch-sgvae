import nltk
import pdb
import grammar
import numpy as np
import h5py
import pickle
import random
import parameters
import grammar_model
from rdkit import Chem
random.seed(42)

params = parameters.load_params()

with open('data/QM9_STAR.pkl', 'rb') as data:
    f = pickle.load(data)
list_df = list(f.loc[:, 'SMILES_GDB-17'])

# list_df[:] = [Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True, canonical=True) for x in list_df if x]

# choosing random ids for train and test
ids = list(range(len(list_df)))
random.shuffle(ids)

chunk = int(0.04 * len(list_df))  # ~ 5k molecules for testing
ids_train = sorted(ids[chunk:])
ids_test = sorted(ids[0:chunk])

L = [list_df[i] for i in ids_train] # smiles for training
L_test = [list_df[i] for i in ids_test] # smiles for test

MAX_LEN = params['max_length']
NCHARS = len(grammar.GCFG.productions())

def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(grammar.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = grammar_model.get_zinc_tokenizer(grammar.GCFG)
    tokens = map(tokenize, smiles)
    parser = nltk.ChartParser(grammar.GCFG)
    parse_trees = [parser.parse(t).__next__() for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot


def main():
    
    OH = np.zeros((len(L),MAX_LEN,NCHARS))
    for i in range(0, len(L), 100):
        print('Processing: i=[' + str(i) + ':' + str(i+100) + ']')
        onehot = to_one_hot(L[i:i+100])
        OH[i:i+100,:,:] = onehot

        
    h5f = h5py.File('data/qm9_grammar_dataset.h5','w')
    h5f.create_dataset('data', data=OH)
    h5f.close()
    
if __name__ == '__main__':
    main()

