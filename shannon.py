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
from typing import List


def is_sorted(numbers: List[str]) -> bool:
    """ Return true iff a list of numbers is in increasing order. """

    return all(a <= b for a, b in zip(numbers[:-1], numbers[1:]))


def compile_cumulatives(probs: List[float]) -> List[float]:
    """ Compute the cumulative probabilities, starting from 0. """

    total = 0.0
    cumulatives = []
    for prob in probs:
        cumulatives.append(total)
        total += prob
    
    return cumulatives


def round_to_binary(number: float, nbits: int, direction: str = "down"):
    """ Round a float to a binary fraction of a given precision. """

    scaled = (2**nbits) * number

    if direction == "down":
        rounded = int(np.floor(scaled))
    elif direction == "up":
        rounded = int(np.ceil(scaled))
    else:
        raise Exception("Unknown direction %r" % direction)

    return np.binary_repr(rounded, width=int(nbits))


def build_shannon_code(probs: List[float]) -> List[str]:
    """ Construct a Shannon code for a sorted list of probabilities. """

    assert is_sorted(probs[::-1])
    
    lengths = np.ceil(-np.log2(probs))
    cumulatives = compile_cumulatives(probs)

    return [round_to_binary(c, w, direction="down")
            for c, w in zip(cumulatives, lengths)]


def build_fano_code(probs: List[float]) -> List[str]:
    """ Construct a Shannon-Fano-Elias code for a distribution. """

    lengths = np.ceil(-np.log2(probs)) + 1
    cumulatives = compile_cumulatives(probs)

    return [round_to_binary(c, w, direction="up")
            for c, w in zip(cumulatives, lengths)]


def is_prefix_free(codewords: List[float]) -> bool:
    """ Return True iff the code satisfies the prefix property. """

    if not codewords:
        return  # an empty list has the prefix property
    
    if any(codewords.count(w) > 1 for w in codewords):
        return False  # a codeword appears twice

    for w1 in codewords:
        for w2 in codewords:
            if w1 == w2:
                continue  # we are comparing a word to itself: OK
            elif len(w1) == len(w2):
                continue  # they have the same length but differ: OK
            elif len(w1) < len(w2):
                if w2.startswith(w1):
                    return False  # w1 is a prefix of w2
            elif len(w2) < len(w1):
                if w1.startswith(w2):
                    return False  # w2 is a prefix of w1
            else:
                raise Exception("Unexpected error at (%r, %r)" % (w1, w2))
    
    return True  # all comparisons above passed


def compute_entropy(probabilities: List[float]) -> float:
    """ Compute the binary entropy of a distribution. """

    return sum(-p*np.log2(p) for p in probabilities if p > 0)


def _test_shannon_coding() -> None:
    """ Test that the Shannon codes are prefix-free and E(N) <= H + 1. """

    for size in [5, 10, 20]:
        for alpha in [0.2, 1.0, 10.]:

            probs = np.random.dirichlet(alpha * np.ones(size))
            probs = sorted(probs, reverse=True)  # in decreasing order
            code = build_shannon_code(probs)
            assert is_prefix_free(code)

            entropy = compute_entropy(probs)
            meanwidth = sum(p*len(w) for p, w in zip(probs, code))
            assert entropy <= meanwidth <= entropy + 1


def _test_fano_coding() -> None:
    """ Test that the Shannon codes are prefix-free and E(N) <= H + 1. """

    for size in [5, 10, 20]:
        for alpha in [0.2, 1.0, 10.]:

            probs = np.random.dirichlet(alpha * np.ones(size))
            code = build_fano_code(probs)
            assert is_prefix_free(code)

            entropy = compute_entropy(probs)
            meanwidth = sum(p*len(w) for p, w in zip(probs, code))
            assert entropy <= meanwidth <= entropy + 2



if __name__ == "__main__":

    assert build_shannon_code([0.6, 0.4]) == ["0", "10"]
    assert build_fano_code([0.6, 0.4]) == ["00", "101"]

    assert build_shannon_code([0.5, 0.5]) == ["0", "1"]
    assert build_fano_code([0.5, 0.5]) == ["00", "10"]

    assert build_shannon_code([0.4, 0.3, 0.2, 0.1]) == ["00", "01", "101", "1110"]
    assert build_fano_code([0.4, 0.3, 0.2, 0.1]) == ["000", "100", "1100", "11101"]
