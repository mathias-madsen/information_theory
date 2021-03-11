"""
ARITHMETIC CODING
-----------------

This script contains some functions pertinent to arithmetic coding, i.e.,
the coding scheme in which a single Shannon-Fano-Elias codeword is chosen
for an entire text using a very careful branching scheme in which the unit
interval is repeatedly divided up in segments whose sizes correspond to
predictive probabilities. Arithmetic coding is described in the 1976 paper
"Generalized Kraft Inequality and Arithmetic Coding" by Jorma Rissanen.

This script was used in the 2018 NASSLLI course on information theory,

    https://www.cmu.edu/nasslli2018/courses/index.html#infotheory

For more information, contact me on mathias@micropsi-industries.com.

                                            Mathias Winther Madsen
                                            Berlin, 3 July 2018
"""

import numpy as np


def find_outer_binary_interval(left, width):
    """ Find the code for a binary interval containing the deciaml interval.

    Arguments:
    ----------
    left : float in [0, 1]
        The left-hand edge of the interval.
    width : float in [0, 1]
        The width of the interval.

    Returns:
    --------
    bits : str
        A string of bits that represents the smallest binary interval that
        contains the inner decimal interval.

    Notes:
    ------
    We test all inequalities using `<=` rather than `<`. This only makes a
    difference for arithmetic coding in case of events of probability zero.

    Examples:
    ---------
    >>> find_outer_binary_interval(0, 1/3)
    '0'
    >>> find_outer_binary_interval(3/4, 1/4)
    '11'
    """

    assert 0 <= left <= 1
    assert 0 <= width <= 1
    
    bits = ""

    while True:

        if left + width <= 1/2:
            bits += "0"
            left = 2*left
            width = 2*width

        elif left >= 1/2:
            bits += "1"
            left = 2*left - 1
            width = 2*width

        else:
            break

    return bits


def find_inner_binary_interval(left, width):
    """ Find the Shannon-Fano-Elias codeword for this decimal interval.

    Arguments:
    ----------
    left : float in [0, 1]
        The left-hand edge of the interval.
    width : float in [0, 1]
        The width of the interval.

    Returns:
    --------
    bits : str
        A string of bits that represents the widest binary interval that
        fits inside of the outer decimal interval.

    Notes:
    ------
    We test all inequalities using `<=` rather than `<`. This only makes a
    difference for arithmetic coding in case of events of probability zero.

    Examples:
    ---------
    >>> find_inner_binary_interval(0, 1/3)
    '00'
    >>> find_inner_binary_interval(3/4, 1/4)
    '11'
    """

    assert 0 <= left <= 1
    assert 0 <= width <= 1

    if left == 0 and width == 1:
        return ""
    
    num_bits = 1

    while True:

        num_steps = 2 ** num_bits
        step_width = 1 / num_steps

        for i in range(num_steps):
            step_left = i * step_width
            if left <= step_left and step_left + step_width <= left + width:
                return np.binary_repr(i, width=num_bits)

        num_bits += 1


def zoom_to_outer_binary_interval(left, width, code):
    """ Represent an inner interval as if its binary container was [0, 1].

    Arguments:
    ----------
    left : float
        The left-hand edge of the initial interval.
    width : float
        The width of the initial interval.
    code : str
        A string of bits representing sequence of bisections and subsequent
        selections of either the lower ('0') or upper ('1') half-interval.

    Returns:
    --------
    left : float
        The left-hand edge of the interval relative to the binary interval
        defined by the bitstring (rather than to relative to [0, 1]).
    width : float
        The width of the interval relative to the binary interval defined by
        the bitstring (rather than relative to the unit interval [0, 1]).

    Examples:
    ---------
    >>> zoom_to_outer_binary_interval(0.0, 0.25, "00")
    (0.0, 1.0)
    >>> zoom_to_outer_binary_interval(0.0, 0.125, "00")
    (0.0, 0.5)
    >>> zoom_to_outer_binary_interval(0.75, 1.00, "1")
    (0.5, 2.0)
    >>> zoom_to_outer_binary_interval(0, 1, "0")
    (0, 2)
    """

    for bit in code:
        if bit == "0":
            left = 2*left
            width = 2*width
        elif bit == "1":
            left = 2*left - 1
            width = 2*width
        else:
            raise Exception("Unexpected bit: %r" % bit)

    return left, width


def find_index_of_outer_decimal(fences, bitstream):
    """ Find the index of the segment containing the binary path.

    Arguments:
    ----------
    fences : iterable of floats
        A sorted array or list of floats that divides the interval between
        fences[0] and fences[-1] into more specific possibilities.
    bits : str
        A binary string representing a binary interval.

    Returns:
    --------
    index : int
        The index of the subinterval (fences[i], fences[i + 1]) that envelops
        the inner binary path, or -1 if no such index exists.
    prefix : str
        A binary string that represents the widest binary interval that can
        be wrapped by one of the fenced sub-intervals.

    Examples:
    ---------
    >>> find_index_of_outer_decimal([0, 1/3, 2/3, 1], "1000")
    (1, '100')
    >>> find_index_of_outer_decimal([0, 1], "0000")
    (0, '')
    >>> find_index_of_outer_decimal([1/2, 1], "0")
    (-1, '')
    """

    for w in range(len(bitstream) + 1):

        substring = bitstream[:w]
        bitint = 0 if not substring else int(bitstream[:w], base=2)
        bitleft = bitint / 2**w
        bitwidth = 1 / 2**w

        for i, (left, right) in enumerate(zip(fences[:-1], fences[1:])):
            if left <= bitleft and bitleft + bitwidth <= right:
                return i, substring

    return -1, ""


def encode(plaintext, model):
    """ Encode the text using a predictive model and an arithmetic encoder.

    Arguments:
    ----------
    plaintext : str
        The text to be encoded.
    model : PredictiveModel
        A PredictiveModel object which must have an .alphabet attribute and
        a .cumulatives method that maps a string to an ordered sequence of
        conditional continuation probabilities, presented in the same order
        as the alphabet.

    Returns:
    --------
    encoding : str
        A binary string that uniquely represents the text.
    
    Notes:
    ------
    This encoder chooses a single Shannon-Fano-Elias codeword that represents
    the entire input text. Divides up the unit interval in tiny segments, each
    of which can then be named using a Shannon-Fano-Elias codeword, which is
    to say, approximated by an inner binary interval.

    This coding scheme in principle requires arbitrary-precision arithmetic.
    We simulate this by zooming on an outer binary interval whenever we can.
    This corresponds to yielding a binary digit whenever it beomes clear that
    the final codeword must start with a given prefix.

    Warning:
    --------
    This enoder has a weakness when it comes to very tiny intervals that
    overlap both with the upper and lower half of the outer interval, as in

                   remaining set of paths
                        \         /
        [-----------------(--|--)-----------------]

         \________ outer binary interval ________/

    Such intervals do no allow us to move the precision-handling from the
    predictive interval (which has limited precision) to the outer interval
    (which has as much precision as the machine's working memory permits),
    since the overlap prevents any further zooming.

    There are ways to address this problem, but they are not implemented here.
    """

    left = np.float64(0)
    width = np.float64(1)

    encoding = ""

    for t, letter in enumerate(plaintext):

       # commit to any bits that have already settled:
        outer_binary = find_outer_binary_interval(left, width)
        left, width = zoom_to_outer_binary_interval(left, width, outer_binary)
        encoding += outer_binary

        # subdivide the remaining interval:
        context = plaintext[:t]
        cumulatives = model.cumulatives(context)
        fences = left + width*cumulatives

        # select the interval corresponding to the next choice:
        i = model.alphabet.index(letter)
        left = fences[i]
        width = fences[i + 1] - fences[i]

     # add bits that have become certain only at the end of the file:
    final_codeword = find_inner_binary_interval(left, width)
    encoding += final_codeword

    print(final_codeword)

    return encoding


def decode(codetext, model):
    """ Decode the text using a predictive model and an arithmetic encoder.

    Arguments:
    ----------
    codetext : str
        A binary codeword for the input plaintext.
    model : PredictiveModel
        A PredictiveModel object which must have an .alphabet attribute and
        a .cumulatives method that maps a string to an ordered sequence of
        conditional continuation probabilities, presented in the same order
        as the alphabet.

    Returns:
    --------
    decoding : str
        A reconstruction of the input plaintext.
    """

    left = np.float64(0)
    width = np.float64(1)

    decoding = ""

    while True:

        outer_binary = find_outer_binary_interval(left, width)
        left, width = zoom_to_outer_binary_interval(left, width, outer_binary)
        codetext = codetext[len(outer_binary):]

        context = decoding
        cumulatives = model.cumulatives(context)
        fences = left + width*cumulatives

        i, codeword = find_index_of_outer_decimal(fences, codetext)

        if i == -1:
            print("leftover", codetext, fences)
            return decoding

        left = fences[i]
        width = fences[i + 1] - fences[i]

        decoding += model.alphabet[i]

        if not codetext:
            return decoding


if __name__ == "__main__":

    from predictive_model import PolyaUrnModel, MarkovModel, MixedModel

##    model = MarkovModel()
    model = MixedModel()
    text = model.sample(100)

    print("Encoding . . .")
    encoded = encode(text, model)
    print("Decoding . . .")
    decoded = decode(encoded, model)
    print("Comparing . . .")
    print(repr(text))
    print(repr(decoded))
    assert text == decoded
    print("Done.\n")

