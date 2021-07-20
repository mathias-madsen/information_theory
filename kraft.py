from typing import List
from typing import Tuple


def build_prefix_code(requested_lengths: List[int],
                      alphabet: Tuple[str] = ("0", "1")) -> List[str]:
    """ Construct a binary prefix code of given code words lengths.
    
    Parameters:
    -----------
    requested_lengths : list of ints >= 0
        A list of requested codeword lengths (repetitions possible).
    alphabet : tuple of strings, default ("0", "1")
        The set of characters out of which to construct the codewords.
    
    Returns:
    --------
    codebook : list of strings
        A list of codewords of the given codeword lengths, sorted by
        increasing order of length.
    
    Raises:
    -------
    ValueError:
        If the list of codeword lengths violate Kraft's inequality.
    
    Notes:
    ------
    This function implements a greedy code construction algorithm which
    builds the prefix shortest-word-first, removing from consideration
    all extensions of a codeword whenever it is added to the codebook.
    """
    
    # complain if Kraft's inequality is violated:
    if sum(len(alphabet) ** (-k) for k in lengths) > 1.0:
        raise ValueError("Codeword lengths %r violate Kraft's inequality."
                         % requested_lengths)

    # we step through the tree of binary strings layer by layer,
    # starting from the root and moving towards longer strings;
    # whenever we use a string as a codeword, we remove all of its
    # descendants from consideration as future codewords.
    allowed_in_current_layer = [""]
    codebook = []
    for current_length in range(max(requested_lengths) + 1):
        # convert prefixes of the current length into codewords
        # until sufficiently many new codewords have been added:
        num_required = sum(k == current_length for k in requested_lengths)
        for _ in range(num_required):
            codebook.append(allowed_in_current_layer.pop(0))
        # extend all remaining prefixes by one more character:
        allowed_in_current_layer = [p + s for p in allowed_in_current_layer
                                    for s in alphabet]
    
    return codebook


if __name__ == "__main__":

    lengths = [1, 1]
    code = build_prefix_code(lengths)
    assert sorted(code) == ["0", "1"]

    lengths = [1, 2, 3, 3]
    code = build_prefix_code(lengths)
    assert sorted([len(w) for w in code]) == lengths

    lengths = [0]
    code = build_prefix_code(lengths)
    assert code == [""]