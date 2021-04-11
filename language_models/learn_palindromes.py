"""
Example training script showing how to learn a palindrome grammar.

The training sentences are sampled from an explicitly given stochastic
context-free grammar over palindromes with the letters 'a' and 'b',
such as 'abba' and 'babab'.

Training takes about the same time as it takes to boil an egg and
usually results in an error rate below one in a thousand -- although
it is a non-convex problem, and there are bad local minima.
"""

import re
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from language_models.grammar import Grammar


def create_random_grammar(num_nonterminals, alphabet):

    N = num_nonterminals
    T = len(alphabet)

    transitions = np.random.gamma(0.1, size=(N, N, N))
    transitions /= np.sum(transitions, axis=(1, 2), keepdims=True)

    emissions = np.random.gamma(0.1, size=(N, T))
    emissions /= np.sum(emissions, axis=1, keepdims=True)

    # make sure that the typical sentence is short:
    transitions *= 0.3
    emissions *= 0.7

    return Grammar(transitions=transitions,
                   emissions=emissions,
                   alphabet=alphabet)


def get_palindrome_grammar(prob_recursive=0.5):
    """ Construct a palindrome grammar.

    `prob_recursive` is the probability of applying the recursive
    rule to the root nonterminal. The expected length of a sentence
    constructed using `K` applications of this rule is `2*K + 0.5`.

    If the probability of applying the recursive rule is higher, the
    language is harder to replicate using regular expressions only,
    making it a more interesting example. However, it will also grow
    the length of the typical sentence, which means that computations
    will be slower.
    """

    ROOT, A, B, C, ROOT_A, ROOT_B, ROOT_C = range(7)

    # even- and odd-length palindromes:
    prob_double = (1 - prob_recursive) / 2
    prob_single = (1 - prob_recursive) / 2

    # only even-length palindromes:
    # prob_double = (1 - prob_recursive)
    # prob_single = 0.0

    palindrome_rules = {

        ROOT: {
            # recurse:
            (A, ROOT_A): prob_recursive / 3,
            (B, ROOT_B): prob_recursive / 3,
            (C, ROOT_C): prob_recursive / 3,
            # terminate in a pair (==> even-length sentence):
            (A, A): prob_double / 3,
            (B, B): prob_double / 3,
            (C, C): prob_double / 3,
            # terminate in a letter (==> odd-length sentence):
            "a": prob_single / 3,
            "b": prob_single / 3,
            "c": prob_single / 3,
        },

        A: {"a": 1.0},
        B: {"b": 1.0},
        C: {"c": 1.0},

        ROOT_A: {(ROOT, A): 1.0},
        ROOT_B: {(ROOT, B): 1.0},
        ROOT_C: {(ROOT, C): 1.0},

    }

    return Grammar(palindrome_rules)


if __name__ == "__main__":

    start_time = time.perf_counter()
    true_grammar = get_palindrome_grammar()
    estimated_grammar = create_random_grammar(2 * len(true_grammar),
                                              true_grammar.alphabet)

    for epochidx in range(100):

        # initialize accumulators with a lot of virtual observations:
        sumtrans = 400. * estimated_grammar.transitions.copy()
        sumemits = 400. * estimated_grammar.emissions.copy()

        print(("*** UPDATE NUMBER %s ***" % (epochidx + 1,)).center(72))
        print()

        losses = []
        reflosses = []
        refgoods = defaultdict(list)
        for _ in tqdm(range(500), leave=False, unit="words"):
            sentence = true_grammar.sample_tree().terminals
            # compute the contributions to the parameter update:
            inner = estimated_grammar.compute_inside_probabilities(sentence)
            outer = estimated_grammar.compute_outside_probabilities(inner, initial=0)
            sumtrans += estimated_grammar.sum_transition_probabilities(inner, outer)
            sumemits += estimated_grammar.sum_emission_probabilities(sentence, outer)
            # for reporting purposes, also compute various loss measures:
            losses.append(-np.log2(inner[0, -1, 0]))
            # also compute statistics related to the true mechanism:
            true_inner = true_grammar.compute_inside_probabilities(sentence)
            reflosses.append(-np.log2(true_inner[0, -1, 0]))
            refgoods[len(sentence)].append(sentence == sentence[::-1])

        model_stats = np.mean(losses), np.std(losses)
        real_stats = np.mean(reflosses), np.std(reflosses)
        print("Mean sentence log-likelihoods (incl. length):")
        print("---------------------------------------------")
        print("  Current model:  %.3f ± %.3f bits." % model_stats)
        print("   Ground truth:  %.3f ± %.3f bits." % real_stats)
        print()

        isgoods = defaultdict(list)
        numgood = 0
        numtotal = 0
        for _ in range(1000):
            snt = estimated_grammar.sample_tree().terminals
            halflen = len(snt) // 2
            numgood += sum(snt[i] == snt[len(snt) - 1 - i] for i in range(halflen))
            numtotal += halflen
            isgoods[len(snt)].append(snt == snt[::-1])
        print("Correctly constrained letters: %s / %s = %.1f pct.\n"
              % (numgood, numtotal, 100.0 * numgood / numtotal))
        print("Number of palindromes:")
        print("----------------------")
        for length in range(2, 16):
            header = "Length %s    --    " % length
            print(header.rjust(25), end="", flush=True)
            numgood = str(sum(isgoods[length])).rjust(4)
            numtotal = str(len(isgoods[length])).ljust(4)
            print("model:  %s / %s    --    " % (numgood, numtotal), end="")
            numgood = str(sum(refgoods[length])).rjust(4)
            numtotal = str(len(refgoods[length])).ljust(4)
            print("ground truth:  %s / %s" % (numgood, numtotal))
        print()

        norms = np.sum(sumtrans, axis=(1, 2)) + np.sum(sumemits, axis=1)
        sumtrans /= norms[:, None, None]
        sumemits /= norms[:, None]
        estimated_grammar.update_from_matrices(sumtrans, sumemits)

        print("Some words sampled from the grammar (after updating):\n")
        for _ in range(30):
            pseudoword = "".join(estimated_grammar.sample_tree().terminals)
            print("%r," % pseudoword, end=" ")
        print(". . .\n")
        print("")

        duration = time.perf_counter() - start_time
        print("Time since start: %02d:%.02d." % np.divmod(duration, 60))
        print()

