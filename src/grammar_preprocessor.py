import nltk
import numpy as np
import torch

from src import grammar
from src.config.hyper_parameters import device
from src.models.SGVAE import Decoder, Encoder


def get_zinc_tokenizer(cfg):
    """
    Creates a tokenizer function for processing SMILES strings based on a given grammar configuration.

    The tokenizer replaces specific long tokens in the SMILES strings with shorter replacements
    and then tokenizes the string into individual characters or tokens.

    Args:
        cfg: The grammar configuration object containing a lexical index of tokens.

    Returns:
        A function that takes a SMILES string as input and returns a list of tokens.

    Raises:
        AssertionError: If the number of long tokens and replacements do not match, or if a replacement
                        token is already present in the lexical index.
    """
    long_tokens = [a for a in list(cfg._lexical_index.keys()) if len(a) > 1]
    replacements = ["$"]
    assert len(long_tokens) == len(replacements)
    for token in replacements:
        assert token not in cfg._lexical_index

    def tokenize(smiles):
        for i, token in enumerate(long_tokens):
            smiles = smiles.replace(token, replacements[i])
        tokens = []
        for token in smiles:
            try:
                ix = replacements.index(token)
                tokens.append(long_tokens[ix])
            except:
                tokens.append(token)
        return tokens

    return tokenize


def pop_or_nothing(S):
    try:
        return S.pop()
    except:
        return "Nothing"


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == "Nothing":
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1 :]
                break
    try:
        return "".join(seq)
    except:
        return ""


class GrammarPreprocessor:
    """
    Test the grammarVAE based on the trained model.
    """

    def __init__(self, parameters):
        self._grammar = grammar
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = get_zinc_tokenizer(self._grammar.GCFG)
        self._n_chars = len(self._productions)
        self._lhs_map = {}
        for ix, lhs in enumerate(self._grammar.lhs_list):
            self._lhs_map[lhs] = ix
        self._params = parameters
        self._encoder = Encoder()
        self._decoder = Decoder()

    def encode(self, smiles):
        """
        Encode a list of smiles strings into the latent space. The input smiles
        is a list of regular smiles which were not used for training.
        """
        assert type(smiles) == list
        tokens = map(self._tokenize, smiles)
        parse_trees = [self._parser.parse(t).__next__() for t in tokens]
        productions_seq = [tree.productions() for tree in parse_trees]
        indices = [
            np.array([self._prod_map[prod] for prod in entry], dtype=int)
            for entry in productions_seq
        ]
        one_hot = np.zeros(
            (len(indices), self._params["input_dim"], self._n_chars), dtype=np.float32
        )
        for i in range(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions), indices[i]] = 1.0
            one_hot[i][np.arange(num_productions, self._params["input_dim"]), -1] = 1.0
        one_hot = torch.from_numpy(one_hot).to(device)  # [batch, MAX_LEN, NUM_OF_RULES]
        one_hot = one_hot.transpose(
            1, 2
        )  # need to reshape to [batch, NUM_OF_RULES, MAX_LEN] for the convolution encoder

        return self._encoder(one_hot)[0]

    def _sample_using_masks(self, unmasked):
        """
        Samples a one-hot vector, masking at each timestep. This is an
        implementation of Algorithm 1 in the paper. Notice that unmasked is a
        torch tensor
        """
        eps = 1e-10
        X_hat = np.zeros_like(unmasked)

        # Create a stack for each input in the batch
        S = np.empty((unmasked.shape[0],), dtype=object)
        for ix in range(S.shape[0]):
            S[ix] = [str(self._grammar.start_index)]

        # Loop over time axis, sampling values and updating masks
        for t in range(unmasked.shape[1]):
            next_nonterminal = [self._lhs_map[pop_or_nothing(a)] for a in S]
            mask = self._grammar.masks[next_nonterminal]
            masked_output = (
                np.exp(unmasked[:, t, :]) * mask.cpu().detach().numpy() + eps
            )  # .cpu().detach().numpy()
            sampled_output = np.argmax(
                np.random.gumbel(size=masked_output.shape) + np.log(masked_output),
                axis=-1,
            )
            X_hat[np.arange(unmasked.shape[0]), t, sampled_output] = 1.0

            # Identify non-terminals in RHS of selected production, and
            # push them onto the stack in reverse order
            rhs = [
                filter(
                    lambda a: (type(a) == nltk.grammar.Nonterminal)
                    and (str(a) != "None"),
                    self._productions[i].rhs(),
                )
                for i in sampled_output
            ]
            for ix in range(S.shape[0]):
                S[ix].extend(list(map(str, rhs[ix]))[::-1])
        return X_hat

    def decode(self, z):
        """Sample from the grammar decoder"""

        unmasked = self._decoder(z)
        unmasked = unmasked.cpu().detach().numpy()

        X_hat = self._sample_using_masks(unmasked)
        # Convert from one-hot to sequence of production rules
        prod_seq = [
            [self._productions[X_hat[index, t].argmax()] for t in range(X_hat.shape[1])]
            for index in range(X_hat.shape[0])
        ]

        return [prods_to_eq(prods) for prods in prod_seq]
