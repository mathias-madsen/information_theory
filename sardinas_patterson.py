"""
This script the Sardinas-Patterson algorithm, which can determine
whether a code is uniquely decodable, and if not, display a string
that has two or more parses.
"""

from collections import defaultdict


def get_compact_parse_representation(sentence, code):
    """ Represent parses of a sentences as dict of forward pointers.
    
    Parameters:
    -----------
    sentence : str
        The sentence to be parsed into words.
    code : set of strings
        The set of available codewords.
    
    Returns:
    --------
    continuations : dict of sets
        For each starting point `s`, the set `continuations[s]`
        contains the set of end points `t` such that

         - the string sentence[s:t] is a code word
         - the string sentence[t:] has at least one parse
        
        The input sentence has a parse if 0 is in `continuations`.
    """

    length = len(sentence)
    continuations = defaultdict(set)
    continuations[length] = []
    for start in reversed(range(len(sentence) + 1)):
        for word in code:
            end = start + len(word)
            if end > length:
                continue  # word too long to fit in
            if sentence[start:end] == word and end in continuations:
                continuations[start].add(end)

    return continuations


def iter_chains(jumpdict, sequence=(0,)):
    """ Yield all complete chains of indices from a {start: end} dict.
    
    For instance, `iter_chains({0: [1], 1: [2], 2: []})` will yield the
    chain `(0, 1, 2)` only, since this stepwise path is the only chain
    from 0 to 2. By contrast, `iter_chains({0: [1, 2], 1: [2], 2: []})`
    will yield the two chains `(0, 2)` and `(0, 1, 2)`, both are feasible
    paths from 0 to 2. (In both cases, 2 is considered a halting state
    because the entry for 2 in the `jumpdict` is empty.)
    """

    start = sequence[-1]
    if not jumpdict[start]:
        yield sequence
    else:
        for end in jumpdict[start]:
            for chain in iter_chains(jumpdict, sequence + (end,)):
                yield chain


def iter_all_parses(sentence, code):
    """ Yield every parse of a sentence as a tuple of code words.
    
    NOTE: the number of parses of a sentence may grow exponentially
    in the length of the sentence, but they can be represented in a
    format that is quadratic in space.
    """
    pointers = get_compact_parse_representation(sentence, code)
    for chain in iter_chains(pointers):
        yield tuple(sentence[start:stop] for start, stop
                    in zip(chain[:-1], chain[1:]))


def find_ambiguous_string(code):
    """ Find an ambiguous string if one exists, otherwise None.
    
    Use the Sardinas-Patterson algorithm to systematically search for
    two sentences that have the same appearence as strings.

    The algorithm looks at pairs of distinct sentences, one always
    bring a prefix of the other, and tries to find out if both of
    them can be completed in such a way that they appear identical.

    Parameters:
    -----------
    code : set of strings
        A set of code words that can be concatenated into strings.

    Returns:
    --------
    ambiguous_string : str or None
        A string with two different parses if one exists, otherwise
        None.

    Note:
    -----    
    For the sake of readability, we do now exhibit two different parses
    of the sentence, although that information could be recovered from
    the algorithm at the cost of some clutter. This means you will have
    to parse the output of this function to actually see the competing
    parses of the ambiguous string.
    """

    # We'll say that a language admits of a string difference D if
    # the strings S and SD are both sentences in the language.
    #
    # We will systematically build up a dict of string differences
    # for the language defined by a code until we have either seen
    # all the string differences, or we have found two distinct
    # sentences that have an empty difference.
    #
    # This is equivalent to traversing all branches of the string
    # tree that are inhabited by at least two sentences.
    
    diffs = dict()
    for shorter in code:
        for longer in code:
            if len(shorter) < len(longer) and longer.startswith(shorter):
                tail = longer[len(shorter):]
                diffs[tail] = shorter

    while True:

        new_diffs = dict()

        for word in code:
            for tail, head in diffs.items():

                # code word AB explains some of the diff ABCD, possibly
                # leaving over some characters CD to be resolved:
                if len(word) <= len(tail) and tail.startswith(word):
                    end_of_tail = tail[len(word):]
                    if end_of_tail not in diffs:
                        new_diffs[end_of_tail] = head + word
                
                # the code word ABCD explains more than the diff AB,
                # thus creating a new diff CD that has to be resolved:
                elif len(word) > len(tail) and word.startswith(tail):
                    end_of_word = word[len(tail):]
                    if end_of_word not in diffs:
                        new_diffs[end_of_word] = head + tail

        # if the empty string is in the new set of diffs, then we
        # have a string that has at least two distinct parses:
        if "" in new_diffs:
            return new_diffs[""]
        
        # if there are no new diffs to discover and we have not
        # get found an ambiguous sentence, then none exist at all:
        if not new_diffs:
            return None

        diffs.update(new_diffs)


if __name__ == "__main__":

    import numpy as np

    def sample_random_code(mean_num_code_words):
        num_param = 1 / mean_num_code_words
        num_words = 1 + np.random.geometric(num_param)
        len_param = 1 / np.log2(mean_num_code_words)
        len_words = np.random.geometric(len_param, size=num_words)
        return set(["".join(np.random.choice(["0", "1"], size=length))
                    for length in len_words])

    codes = [
        {"0", "1"},
        {"01", "10", "011"},
        {"00", "11", "0110"},
        {"01", "10", "0110"},
        {"01", "1110", "101"},
        {"0", "10", "100"},
        {"011", "01110", "1", "10011", "1110"},
        sample_random_code(2),
        sample_random_code(3),
        sample_random_code(4),
        sample_random_code(5),
        sample_random_code(6),
    ]

    for code in codes:
        string = find_ambiguous_string(code)
        if string is None:
            print("Code %s is uniquely decodable." % code)
            print()
        else:
            print("String %r is ambiguous under %r." % (string, code))
            print("Parses: %r" % list(iter_all_parses(string, code)))
            print()
