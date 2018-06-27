"""
HUFFMAN CODING DEMO
-------------------

This script constructs a binary Huffman code for the coin flipping sequences
of length n, and then visualizes that code as a tree. It was used in lecture 2
of the 2018 NASSLLI course on information theory.

For more information, see the course website on

    https://www.cmu.edu/nasslli2018/courses/index.html#infotheory

or contact me on mathias@micropsi-industries.com.

                                            Mathias Winther Madsen
                                            Pittsburgh, 27 June 2018
"""

import numpy as np
from scipy.special import entr  # binary entropy in base e
import itertools
from collections import OrderedDict
import matplotlib
from matplotlib import pyplot as plt

from huffman import build_huffman_tree, iter_codewords


# edit the TeX font displayed by matplotlib:
matplotlib.rcParams["font.size"] = 24
matplotlib.rcParams["font.monospace"] = "Courier New"
matplotlib.rcParams["mathtext.fontset"] = "cm"


def size(tuple_of_tuples):
    " Recursively compute the number of children of a nested tuple. "

    if type(tuple_of_tuples) != tuple:
        return 1

    return sum(size(element) for element in tuple_of_tuples)


def replace_leaves(tree, replacements, _next_int=0, _recursive=False):
    """ Replace the leaves of the tree by the given elements, depth-first.

    Arguments:
    ----------
    tree : tuple or object
        A leaf, tree, or tree branch whose leves we wish to replace. Its
        leaves can be arbitrary objects (since they will be replaced)
    replacements : iterable
        A list of other iterable of elements we wish to put in place of
        the original leaves of the tree.
    _next_int : int
        The index of the next unused element on the list of replacements.
        This argument is passed on to the right-hand branch of a tree to
        inform it how many replacements the left-hand branch consumed.
    _recursive : bool
        True if the current call to the function was requested by the
        function itself. This argument is used to block the top call
        from returning the `_next_intz` counter.

    Returns:
    --------
    tree : tuple
        A new tree whose leaves are `replacements`. The leaves of the input
        tree will be replaced in a left-to-right, depth-first manner.
    """

    if type(tree) != tuple:
        return replacements[_next_int], _next_int + 1

    else:
        new_branches = []
        for branch in tree:
            args = branch, replacements, _next_int, True
            new_branch, _next_int = replace_leaves(*args)
            new_branches.append(new_branch)
        if _recursive:
            return tuple(new_branches), _next_int
        else:
            return tuple(new_branches)


def draw_connections(tree, leftshift=0.3):
    """ Draw a binary tree whose leaves are (x, y)-positions.

    Arguments:
    ----------
    tree : tuple or numpy array
        A tree whose leaves are numpy arrays [ x  y ], or such a leaf.
    leftshift : float
        The minimal distance (in axis units) by which a parent node
        should be separated from its children.

    Returns:
    --------
    xy : numpy array
        The (x, y)-location of the root of the tree.
    n : int
        The number of leaves of the tree.
    """

    if type(tree) != tuple:

        return tree, 1

    else:

        xy1, n1 = draw_connections(tree[0])
        xy2, n2 = draw_connections(tree[1])
        
        xy = (n1*xy1 + n2*xy2) / (n1 + n2)
        xy[0] = xy1[0] - n2*leftshift

        plt.arrow(*xy, *(xy1 - xy))
        plt.arrow(*xy, *(xy2 - xy))

        return xy, n1 + n2
                
    
def get_coin_flipping_distribution(n=4, p=0.2):
    """ Return the distribution over coin flipping sequences of length n.

    Arguments:
    ----------
    n : int >= 0
        The sequence length
    p : float in [0, 1]
        The bias of the coin.

    Returns:
    --------
    distribution : dict
        A sequence distribution in the format {sequence: probability}.
        The sequences are represented as strings, e.g., '0010111010'.
    """

    distribution = {}

    for bits in itertools.product("01", repeat=n):
        k = sum(int(bit) for bit in bits)
        logp = k*np.log(p) + (n - k)*np.log(1 - p)
        distribution["".join(bits)] = np.exp(logp)

    return distribution


def draw_coin_flipping_tree(n=4, p=0.2):
    """ Construct and draw a Huffman code for a sequence of n coin flips.

    Arguments:
    ----------
    n : int >= 0
        The sequence length
    p : float in [0, 1]
        The bias of the coin.
    """

    distribution = get_coin_flipping_distribution(n=n, p=p)

    tree = build_huffman_tree(distribution)
    code = OrderedDict(iter_codewords(tree))
    codesize = len(code)

    H = entr(p) / np.log(2)  # convert to base 2
    EK = sum(distribution[letter] * len(code[letter])
             for letter in distribution)

    min_y = -1
    max_y = +1
    
    max_x = 1.0
    
    locations = max_x * np.ones((codesize, 2))
    locations[:, 1] = np.linspace(min_y, max_y, codesize)
    location_tree = replace_leaves(tree, locations)

    figure = plt.figure(figsize=(12, 8))

    (min_x, middle_y), num_children = draw_connections(location_tree)
    treewidth = 1 - min_x

    bigfontsize = matplotlib.rcParams["font.size"]
    smallfontsize = bigfontsize / np.log2(1 + n**0.5)
    fontargs = dict(fontsize=smallfontsize,
                    family="monospace",
                    fontweight="bold")

    for (x, y), codeword in zip(locations, code.values()):
        plt.text(x, y, " " + codeword, ha="left", va="center", **fontargs)

    titlestring = "$n=%s,\\quad{}H/n=%.3f,\\quad{}E(K/n)=%.3f$" % (n, H, EK/n)
    plt.title(titlestring, fontsize=bigfontsize)

    plt.xlim(min_x - 0.1*treewidth,
             max_x + 0.3*treewidth)

    plt.ylim(min_y - 0.2,
             max_y + 0.3)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(figure)


if __name__ == "__main__":

    p = 0.1
    for n in range(1, 8):
        draw_coin_flipping_tree(n=n, p=p)
