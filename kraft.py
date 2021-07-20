from typing import List
from typing import Tuple


def build_prefix_code(requested_lengths: List[int],
                      alphabet: Tuple[str] = ("0", "1")) -> List[str]:
    """ Construct a binary prefix code of given codeword lengths.
    
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
    builds the prefix code shortest-word-first, adding available codewords
    as necessary, and removing all possible extensions of a codeword
    whenever it is added to the codebook.
    """
    
    # complain if Kraft's inequality is violated:
    if sum(len(alphabet) ** (-k) for k in lengths) > 1.0:
        raise ValueError("Codeword lengths %r violate Kraft's inequality."
                         % requested_lengths)

    # we step through the tree of binary strings by increasing length;
    # whenever we use a string as a codeword, we remove its descendants
    # from consideration as future codewords:
    allowed_at_current_length = [""]
    codebook = []
    for current_length in range(max(requested_lengths) + 1):
        # add the required number of codewords at this length
        # by cutting them out of the list of available prefixes:
        num_required = sum(k == current_length for k in requested_lengths)
        for _ in range(num_required):
            codebook.append(allowed_at_current_length.pop(0))
        # extend all remaining prefixes by one more character:
        allowed_at_current_length = [p + s for p in allowed_at_current_length
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