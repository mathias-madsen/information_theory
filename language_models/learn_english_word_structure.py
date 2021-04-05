import re
import numpy as np
from tqdm import tqdm

from language_models.grammar import Grammar


def create_random_grammar(num_nonterminals=4):

    alphabet = tuple("abcdefghijklmnopqrstuvwxyz")

    N = num_nonterminals
    T = len(alphabet)

    transitions = np.random.gamma(1.0, size=(N, N, N))
    transitions /= np.sum(transitions, axis=(1, 2), keepdims=True)
    
    emissions = np.random.gamma(1.0, size=(N, T))
    emissions /= np.sum(emissions, axis=1, keepdims=True)

    # make sure that the typical sentence is short:
    transitions *= 0.25
    emissions *= 0.75

    return Grammar(transitions=transitions,
                   emissions=emissions,
                   alphabet=alphabet)


def extract_words(text):

    word_pattern = """
        \s  # whitespace before
        ["']?  # possibly quotation marks
        ([a-z]+)  # string of lowercase letters
        ['".!?]{0,2}  # possibly quotation marks and punctuation
        \s    # whitespace after
        """

    return re.findall(word_pattern, text, re.VERBOSE)


def split_list(elements, min_chunk_size=1000):
    """ Split a list into smaller arrays of a given length.
    
    If the length of the input length is not perfectly divisible with
    the requested chunk length, the dangling elements are appended to
    the final chunk, which may therefore be up to twice as long.
    """
    
    num_chunks = len(elements) // min_chunk_size
    fences = min_chunk_size * np.arange(1, num_chunks)

    return np.split(elements, fences)


if __name__ == "__main__":

    with open("long_text.txt", "r") as source:
        words = extract_words(source.read())
    
    grammar = create_random_grammar(20)

    for updateidx, chunk in enumerate(split_list(words, 1000)):

        print("--- Epoch %s ---" % (updateidx + 1))

        # initialize accumulators with a few virtual observations:
        sumtrans = 10.0 * grammar.transitions.copy()
        sumemits = 10.0 * grammar.emissions.copy()

        losses = []
        for sentence in tqdm(chunk, leave=False):
            inner = grammar.compute_inside_probabilities(sentence)
            outer = grammar.compute_outside_probabilities(inner, initial=0)
            sumtrans += grammar.sum_transition_probabilities(inner, outer)
            sumemits += grammar.sum_emission_probabilities(sentence, outer)
            loglike = np.log(inner[0, -1, 0])
            losses.append((-1) * loglike / len(sentence))

        print("Mean loss per character (before updating):")
        print("%.3f ± %.3f bits.\n" % (np.mean(losses), np.std(losses)))

        print("Some words sampled from the grammar (before updating):\n")
        for _ in range(30):
            pseudoword = "".join(grammar.sample_tree().terminals)
            print("%r," % pseudoword, end=" ")
        print(". . .\n")

        norms = np.sum(sumtrans, axis=(1, 2)) + np.sum(sumemits, axis=1)
        sumtrans /= norms[:, None, None]
        sumemits /= norms[:, None]
        grammar.update_from_matrices(sumtrans, sumemits)

    print("Some samples from the estimated grammar:")
    print()
    for _ in range(10):
        print("".join())
    print()
