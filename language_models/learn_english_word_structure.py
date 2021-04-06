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

    chunk_size = 1000

    with open("long_text.txt", "r") as source:
        words = extract_words(source.read())
        wordcounts = dict(zip(*np.unique(words, return_counts=True)))
        words = [w for w in words if wordcounts[w] > 10]  # remove freaks
        # words = list(set(words))  # do not repeat frequent words
        # words = [w for w in words if 5 <= len(w) <= 20]

    letters, counts = np.unique(list("".join(words)), return_counts=True)
    counts += 100  # add virtual observations of each letter
    freqs = counts / np.sum(counts)
    freqdict = {ch: num / np.sum(counts) for ch, num in zip(letters, counts)}
    baseline = lambda word: sum(-np.log2(freqdict[c]) for c in word)

    chunks = split_list(words, chunk_size)
    num_chunks = len(chunks)
    chunks = [chunks[i] for i in np.random.permutation(num_chunks)]

    print("Loaded a data set of %s words.\n" % len(words))

    print("The data was split up into %s chunks of size %s.\n"
          % (num_chunks, chunk_size))

    print("Example words from the data set:\n")
    print(", ".join("%r" % w for w in np.random.choice(words, size=50)))
    print("\n")

    grammar = create_random_grammar(10)

    for epochidx in range(10):

        grammar.save_as("wordgrammar.npz")

        for updateidx, chunk in enumerate(chunks):

            print("--- Epoch %s, step %s / %s ---\n" %
                  (epochidx + 1, updateidx + 1, len(chunks)))

            # initialize accumulators with a few virtual observations:
            sumtrans = 0.01 * len(chunk) * grammar.transitions.copy()
            sumemits = 0.01 * len(chunk) * grammar.emissions.copy()
            sumemits += 1e-5 * len(chunk) * freqs  # fallback: letter freqs

            relative_losses = []
            monogram_losses = []
            losses = []

            for sentence in tqdm(chunk, leave=False, unit="words"):
                # compute the contributions to the parameter update:
                inner = grammar.compute_inside_probabilities(sentence)
                outer = grammar.compute_outside_probabilities(inner, initial=0)
                sumtrans += grammar.sum_transition_probabilities(inner, outer)
                sumemits += grammar.sum_emission_probabilities(sentence, outer)
                # for reporting purposes, also compute various loss measures:
                loss = (-1) * np.log(inner[0, -1, 0])  # in absolute terms
                losses.append(loss / len(sentence))  # / uniform-model loss
                monogram_loss = baseline(sentence)
                monogram_losses.append(monogram_loss / len(sentence))
                relative_losses.append(loss / monogram_loss)

            stats = np.mean(losses), np.std(losses)
            monostats = np.mean(monogram_losses), np.std(monogram_losses)
            increasing = list(chunk[np.argsort(relative_losses)])

            print("Model loss per on this batch: %.3f ± %.3f bits." % stats)
            print("Baseline loss per on this batch: %.3f ± %.3f bits." % monostats)
            print("Highest excessive loss: %r" % increasing[-5:])
            print("Lowest excessive loss: %r" % increasing[:5])
            print("")

            norms = np.sum(sumtrans, axis=(1, 2)) + np.sum(sumemits, axis=1)
            sumtrans /= norms[:, None, None]
            sumemits /= norms[:, None]
            grammar.update_from_matrices(sumtrans, sumemits)

            print("Some words sampled from the grammar (after updating):\n")
            for _ in range(30):
                pseudoword = "".join(grammar.sample_tree().terminals)
                print("%r," % pseudoword, end=" ")
            print(". . .\n")
            print("")
