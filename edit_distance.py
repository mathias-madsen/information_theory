import numpy as np


def edit_distance_analysis(word1, word2):
    """" Obtained details on how to convert one word into the other.

    Notes:
    ¯¯¯¯¯¯
    The edit commands considered are:

     - correct transmission of one letter (cost=0)
     - spurious deletion from the input stream (cost=1)
     - spurious insertion into the output stream (cost=1)
     - corrupted transmission of one letter (cost=1)
     - transposed transmission of two letters (cost=1)

    Parameters:
    ¯¯¯¯¯¯¯¯¯¯¯
    word1 : str
        The first word, nominally considered the observed "input stream."
    word1 : str
        The second word, nominally considered the observed "output stream."

    Returns:
    ¯¯¯¯¯¯¯¯
    distance matrix : array of shape (len(word1) + 1, len(word2) + 1)
        A matrix of edit distances recording in entry (i, j) the smallest
        number of edits necessary to convert the string word1[:i] into
        the string word2[:j].
    commands : list of strings
        An minimal and ordered list of commands that, when executed, will
        convert word1 into word2. When there are several optimal programs
        that achieve this goal, an arbitrary selection is made among them.
    """

    n = len(word1) + 1
    m = len(word2) + 1

    dist = np.zeros((n, m), dtype=np.uint64)
    cmds = dict()

    dist[0, 0] = 0
    cmds[0, 0] = []

    for i in range(1, n):
        dist[i, 0] = i  # i deletions
        cmds[i, 0] = cmds[i - 1, 0] + ["delete %s from input" % word1[i - 1]]
    for j in range(1, m):
        dist[0, j] = j  # j insertions
        cmds[0, j] = cmds[0, j - 1] + ["insert %s into output" % word2[j - 1]]

    for i in range(1, n):
        for j in range(1, m):

            deletion_cost = dist[i - 1, j] + 1
            insertion_cost = dist[i, j - 1] + 1
            substitution_increment = 0 if word1[i - 1] == word2[j - 1] else 1
            substitution_cost = dist[i - 1, j - 1] + substitution_increment
            possible_costs = [deletion_cost, insertion_cost, substitution_cost]

            if i > 1 and j > 1:
                transposition_cost = dist[i - 2, j - 2] + 1
                pair1 = word1[i - 2 : i]
                pair2 = word2[i - 2 : i]
                if pair1 == pair2[::-1]:
                    possible_costs.append(transposition_cost)

            winner = np.argmin(possible_costs)
            dist[i, j] = possible_costs[winner]

            if winner == 0:
                command = "delete %s from input" % word1[i - 1]
                cmds[i, j] = cmds[i - 1, j] + [command]

            elif winner == 1:
                command = "insert %s into output" % word2[j - 1]
                cmds[i, j] = cmds[i, j - 1] + [command]

            elif winner == 2 and substitution_increment == 0:
                command = "transmit %s" % word1[i - 1]
                cmds[i, j] = cmds[i - 1, j - 1] + [command]

            elif winner == 2 and substitution_increment == 1:
                command = "corrupt %s into %s" % (word1[i - 1], word2[j - 1])
                cmds[i, j] = cmds[i - 1, j - 1] + [command]

            elif winner == 3:
                command = "transpose %s" % (word1[i - 2 : i])
                cmds[i, j] = cmds[i - 2, j - 2] + [command]

            else:
                raise Exception("Unexpected argmin: %s" % winner)

    return dist, cmds[n - 1, m - 1]


def edit_distance(word1, word2):
    """ Compute the number of edits separating the two words.

    Notes:
    ¯¯¯¯¯¯
    The edit commands considered are:

     - correct transmission of one letter (cost=0)
     - spurious deletion from the input stream (cost=1)
     - spurious insertion into the output stream (cost=1)
     - corrupted transmission of one letter (cost=1)
     - transposed transmission of two letters (cost=1)

    Returns:
    ¯¯¯¯¯¯¯¯
    distance : uint64
        The smallest number of edits necessary to convert word1 into word2.
    """

    matrix, commands = edit_distance_analysis(word1, word2)

    return matrix[-1, -1]


if __name__ == "__main__":

    testwords = "AB BA CATS BATS STAB TABS".split(" ")
    testwords.append("")
    testwords.append("A")

    for word1 in testwords:
        for word2 in testwords:
            distance, recipe = edit_distance_analysis(word1, word2)
            print((word1, word2), distance[-1, -1])
            print(recipe)
            print()
