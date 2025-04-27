"""
This module defines the context-free grammar (CFG) and related data structures for the Grammar Variational Autoencoder (Grammar VAE).

Key concepts:
- The grammar is defined in Backus-Naur Form (BNF) and parsed using NLTK's CFG tools.
- The production rules, left-hand side (LHS) and right-hand side (RHS) symbols, and masks are used to constrain the decoder during generation.
- The masks ensure that at each decoding step, only valid production rules for the current non-terminal can be chosen.

References:
- Grammar Variational Autoencoder: https://arxiv.org/abs/1703.01925
"""

import nltk
import numpy as np
import six
import torch
from src.config.hyper_parameters import device

# the zinc grammar
gram = """smiles -> chain
atom -> bracket_atom
atom -> aliphatic_organic
atom -> aromatic_organic
aliphatic_organic -> 'C'
aliphatic_organic -> 'N'
aliphatic_organic -> 'O'
aliphatic_organic -> 'F'
aromatic_organic -> 'c'
aromatic_organic -> 'n'
aromatic_organic -> 'o'
bracket_atom -> '[' BAI ']'
BAI -> isotope symbol BAC
BAI -> symbol BAC
BAI -> isotope symbol
BAI -> symbol
BAC -> chiral BAH
BAC -> BAH
BAC -> chiral
BAH -> hcount BACH
BAH -> BACH
BAH -> hcount
BACH -> charge
symbol -> aliphatic_organic
symbol -> aromatic_organic
isotope -> DIGIT
isotope -> DIGIT DIGIT
isotope -> DIGIT DIGIT DIGIT
DIGIT -> '1'
DIGIT -> '2'
DIGIT -> '3'
DIGIT -> '4'
DIGIT -> '5'
DIGIT -> '6'
DIGIT -> '7'
DIGIT -> '8'
chiral -> '@'
chiral -> '@@'
hcount -> 'H'
hcount -> 'H' DIGIT
charge -> '-'
charge -> '-' DIGIT
charge -> '-' DIGIT DIGIT
charge -> '+'
charge -> '+' DIGIT
charge -> '+' DIGIT DIGIT
bond -> '-'
bond -> '='
bond -> '#'
bond -> '/'
bond -> '\\'
ringbond -> DIGIT
ringbond -> bond DIGIT
branched_atom -> atom
branched_atom -> atom RB
branched_atom -> atom BB
branched_atom -> atom RB BB
RB -> RB ringbond
RB -> ringbond
BB -> BB branch
BB -> branch
branch -> '(' chain ')'
branch -> '(' bond chain ')'
chain -> branched_atom
chain -> chain branched_atom
chain -> chain bond branched_atom
Nothing -> None"""

# form the CFG and get the start symbol
GCFG = nltk.CFG.fromstring(gram)  # Parse the grammar string into an NLTK CFG object
# GCFG.productions() returns a list of all production rules in the grammar.
# GCFG.productions()[0] is the first production rule (the start rule: 'smiles -> chain').
# .lhs() returns the left-hand side (LHS) non-terminal of the production rule.
start_index = GCFG.productions()[0].lhs()  # The start symbol of the grammar

# collect all lhs symbols, and the unique set of them
# a.lhs().symbol() gets the string name of the LHS non-terminal for each production rule
all_lhs = [a.lhs().symbol() for a in GCFG.productions()]
lhs_list = []
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)

D = len(GCFG.productions())

# this map tells us the rhs symbol indices for each production rule
# For each production rule, we collect the indices of RHS symbols that are non-terminals.
# This is used to track which non-terminals are expanded at each step in the parse tree.
rhs_map = [None]*D
count = 0
for a in GCFG.productions():
    rhs_map[count] = []
    for b in a.rhs():
        # a.rhs() returns a tuple of symbols on the right-hand side of the production rule.
        # If b is a non-terminal (not a string), get its symbol and find its index in lhs_list.
        if not isinstance(b,six.string_types):
            s = b.symbol()
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

# masks is a binary matrix of shape (num_lhs, num_productions)
# masks[i, j] == 1 if the j-th production rule has lhs == lhs_list[i]
# This is used to mask out invalid production rules during decoding in the VAE.
masks = np.zeros((len(lhs_list),D))
count = 0

# this tells us for each lhs symbol which productions rules should be masked
for sym in lhs_list:
    # is_in is a binary vector indicating which productions have lhs == sym
    is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1,-1)
    masks[count] = is_in
    count = count + 1

# index_array[i] gives the index of the lhs symbol for the i-th production rule
# This is used to quickly map a production rule to its lhs index
index_array = []
for i in range(masks.shape[1]):
    index_array.append(np.where(masks[:,i]==1)[0][0])
ind_of_ind = np.array(index_array)
ind_of_ind = torch.Tensor(ind_of_ind).to(device)

# max_rhs is the maximum number of non-terminals on the RHS of any production rule
max_rhs = max([len(l) for l in rhs_map])
# Convert masks to a torch ByteTensor for use in the VAE decoder
masks = torch.ByteTensor(masks).to(device)
