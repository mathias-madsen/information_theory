"""
SHANNON-FANO CODING
-------------------

This script contains some functions pertinent to Shannon-Fano coding,
i.e., the coding scheme in which cumulative probabilities are used as
codewords.

Shannon-Fano coding is described by Shannon in section 9 of his paper
"A Mathematical Theory of Communication" (Bell Systems Technical Journal,
1948), where it is provided as an alternative proof of his "fundamental
theorem for a noiseless channel" (the source coding theorem).

This script was used in the 2018 NASSLLI course on information theory,

    https://www.cmu.edu/nasslli2018/courses/index.html#infotheory

For more information, contact me on mathias@micropsi-industries.com.

                                            Mathias Winther Madsen
                                            Pittsburgh, 27 June 2018
"""

import numpy as np


def build_shannon_code(distribution):
    """ Construct the Shannon code implied by the given distribution.

    Arguments:
    ----------
    distribution : dict
        A letter distribution in the format {letter: probability}.

    Returns:
    --------
    code : dict
        A table of codewords in the format {letter: codeword}.
    """

    # Sort letters in decreasing order of probability:
    letters = sorted(distribution, key=lambda letter: -distribution[letter])

    probabilities = [distribution[letter] for letter in letters]
    cumulative = np.cumsum([0] + probabilities)

    ideal_codeword_widths = -np.log2(probabilities)
    actual_widths = [int(width) for width in np.ceil(ideal_codeword_widths)]

    codewords = []
    for left_edge, width in zip(cumulative, actual_widths):
        left_edge_rounded_down = int(2**width * left_edge)
        codeword = np.binary_repr(left_edge_rounded_down, width=width)
        codewords.append(codeword)

    return {letter: codeword for letter, codeword in zip(letters, codewords)}


def verify_codewords(codewords):
    """ Verify that a list of codewords has the prefix property. """

    if not codewords:
        return  # an empty list has the prefix property
    
    codewords = list(codewords)  # in case they are a dict_keys object
    bits = set("".join(codewords))  # output alphabet
    assert bits == set("01") or bits == set("0") or bits == set("1")
    
    for i in range(len(codewords)):
        for j in range(len(codewords)):
            if i == j:
                continue
            wi = codewords[i]
            wj = codewords[j]
            assert not wi.startswith(wj), (wj, wi)
            assert not wj.startswith(wi), (wi, wj)

    # verify that the codewords satisfy Kraft's inequality:
    assert sum(0.5 ** len(codeword) for codeword in codewords) <= 1, codewords


def entropies(distribution, code):
    """ Compute the expected codeword length, and the entropy bound.

    Arguments:
    ----------
    distribution : dict
        A distribution over letters in the format {letter: probability}.
    code : dict
        A table of codewords in the format {letter: codeword}.

    Returns:
    --------
    true_entropy : float >= 0
        The entropy of the letter distribution, in bits.
    expected_codeword_length : float >= 0
        The mean number of bits used to encode letters from the given
        distribution when using the given code.
    """

    entropy = sum(-p*np.log2(p) for p in distribution.values() if p > 0)
    mean_width = sum(p*len(code[a]) for a, p in distribution.items())
    
    return entropy, mean_width


if __name__ == "__main__":

    for size in [5, 10, 20]:
        for alpha in [0.2, 1.0, 10.]:

            alphabet = [chr(i) for i in range(65, 91)]
            letters = np.random.choice(alphabet, size=size, replace=False)
            probs = np.random.dirichlet(alpha * np.ones(size))
            dist = {letter: prob for letter, prob in zip(letters, probs)}

            code = shannon_code(dist)
            verify_codewords(code.values())
            
            entropy, mean_width = entropies(dist, code)
            assert entropy <= mean_width
            assert entropy + 2 >= mean_width
