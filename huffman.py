"""
HUFFMAN CODING
--------------

This script contains some functions pertinent to Huffman coding, that is,
the coding scheme in which a binary tree (and thus a prefix code) is built
bottom-up by repeatedly joining together the two least likely input symbols.
Huffman coding was first described by David A. Huffman in "A Method for the
Construction of Minimum-Redundancy Codes" (Proceedings of the IRE, 1952)

This script was used in the 2018 NASSLLI course on information theory,

    https://www.cmu.edu/nasslli2018/courses/index.html#infotheory

For more information, contact me on mathias@micropsi-industries.com.

                                            Mathias Winther Madsen
                                            Pittsburgh, 27 June 2018
"""

from collections import OrderedDict


def build_huffman_tree(distribution):
    """ From a dict of letter probabilities, create a Huffman tree.

    Arguments:
    ----------
    distribution : dict
        A distribution in the form {letter: probability}

    Returns:
    --------
    tree : tuple
        A Huffman tree generated from the letter distributions.
        The tree is a tuple of tuples of tuples of ... of letters;
        or speaking recursively, a tree is either a letter or a
        pair of two trees.
    """

    distribution = distribution.copy()  # since we will edit the dict
    
    while len(distribution) > 1:

        A = min(distribution.keys(), key=lambda k: distribution[k])
        prob_A = distribution.pop(A)

        B = min(distribution.keys(), key=lambda k: distribution[k])
        prob_B = distribution.pop(B)

        distribution[A, B] = prob_A + prob_B

    root, = distribution.keys()

    return root


def iter_codewords(tree):
    """ Iterate over the codewords implied by a binary tree.

    Arguments:
    ----------
    tree : tuple or string
        A string (a terminal node) or a pair of trees.

    Yields:
    -------
    letter : str
        An input symbol to be encoded.
    codeword : str
        The corresponding output code.
    """

    if type(tree) is not tuple:
        yield tree, ""

    else:
        for prefix, branch in enumerate(tree):
            for codeword, suffix in iter_codewords(branch):
                yield codeword, str(prefix) + suffix


def build_huffman_code(distribution):
    """ Choose a Huffman code for the given distribution dict.

    Arguments:
    ----------
    distribution : dict
        A letter distribution in the format {letter: prob}.

    Returns:
    --------
    code : OrderedDict
        A table of codewords in the format {letter: codeword}.
    """

    tree = build_huffman_tree(distribution)
    code = OrderedDict(iter_codewords(tree))
    
    return code


def sample_random_letter_distribution(concentration=1.):
    """ Construct a random probability distribution over {A, B, ..., Z}.

    Arguments:
    ----------
    concentration : float >= 0
        A parameter of the prior from which we sample the vector of letter
        probabilities. Higher values mean that the distribution will be
        more likely to place all its mass at a single letter. Values close
        to 0 will yield near-uniform distributions.

    Returns:
    --------
    distribution : dict
        A letter distribution in the format {letter: probability}.
    """

    import numpy as np
    
    alphabet = [chr(i) for i in range(65, 91)]
    alpha = 1/concentration * np.ones(len(alphabet))
    probabilities = np.random.dirichlet(alpha)

    return {letter: prob for letter, prob in zip(alphabet, probabilities)}


if __name__ == "__main__":

    # an example:
    
    distribution = sample_random_letter_distribution()
    code = build_huffman_code(distribution)

    lines = [("%r  %.3f  %r" % (letter, prob, code[letter]))
             for letter, prob in distribution.items()]

    print("\n".join(lines), "\n")
