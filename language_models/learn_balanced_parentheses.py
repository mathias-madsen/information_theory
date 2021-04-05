import numpy as np
from tqdm import tqdm

from language_models.grammar import Grammar


start, left, mid, right = range(4)

true_rulebooks = {

    start: {
        (left, right): 0.5,  # end here with a pair of parentheses
        (left, mid): 0.3,  # open a pair, with nesting possible
        (start, start): 0.2  # insert two pairs, possibly nested
        },

    mid: {
        (start, right): 1.0,  # insert child and close debt
        },

    left: {+1: 1.0},  # open a parenthesis = increase nesting depth
    right: {-1: 1.0},  # close a parenthesis = decrease nesting depth

    }


def check(increments):
    """ Check paren nesting is non-negative and closes at 0. """

    if not all(incr in [-1, +1] for incr in increments):
        return False

    depths = np.cumsum(increments)
    never_close_unopened = np.all(depths >= 0)
    eventually_close_all = depths[-1] == 0

    return never_close_unopened and eventually_close_all


def create_random_grammar(num_nonterminals=4):

    N = num_nonterminals

    transitions = np.random.gamma(1.0, size=(N, N, N))
    transitions /= np.sum(transitions, axis=(1, 2), keepdims=True)
    
    emissions = np.random.gamma(1.0, size=(N, 2))
    emissions /= np.sum(emissions, axis=1, keepdims=True)

    # make sure that the typical sentence is short:
    transitions *= 0.25
    emissions *= 0.75

    return Grammar(transitions=transitions,
                   emissions=emissions,
                   alphabet=[+1, -1])


if __name__ == "__main__":

    true_grammar = Grammar(true_rulebooks)
    estimated_grammar = create_random_grammar()

    for updateidx in range(30):

        print("--- Epoch %s ---" % (updateidx + 1))

        # initialize accumulators with a few virtual observations:
        sumtrans = 0.5 * estimated_grammar.transitions.copy()
        sumemits = 0.5 * estimated_grammar.emissions.copy()

        losses = []
        for _ in tqdm(range(400), leave=False):
            tree = true_grammar.sample_tree()
            sentence = tree.terminals
            if len(sentence) > 25:
                continue  # accept slight bias to speed up things a lot
            assert check(sentence)  # for sanity, verify the true grammar
            inner = estimated_grammar.compute_inside_probabilities(sentence)
            outer = estimated_grammar.compute_outside_probabilities(inner, initial=0)
            sumtrans += estimated_grammar.sum_transition_probabilities(inner, outer)
            sumemits += estimated_grammar.sum_emission_probabilities(sentence, outer)
            loglike = np.log(inner[0, -1, 0])
            losses.append((-1) * loglike / len(sentence))

        statsloss = np.mean(losses), np.std(losses)
        print("Mean loss per character: %.3f ± %.3f bits." % statsloss)

        trees = [estimated_grammar.sample_tree() for _ in range(400)]
        balanced = [check(tree.terminals) for tree in trees]
        proportion = sum(balanced), len(balanced)
        print("Proportion balanced sentences: %s / %s" % proportion)
        print("")

        norms = np.sum(sumtrans, axis=(1, 2)) + np.sum(sumemits, axis=1)
        sumtrans /= norms[:, None, None]
        sumemits /= norms[:, None]
        estimated_grammar.update_from_matrices(sumtrans, sumemits)

    print("Some samples from the true grammar:")
    print()
    for _ in range(10):
        terms = true_grammar.sample_tree().terminals
        string = "".join("_[]"[incr] for incr in terms)
        print(string)
    print()

    print("Some samples from the estimated grammar:")
    print()
    for _ in range(10):
        terms = estimated_grammar.sample_tree().terminals
        string = "".join("_[]"[incr] for incr in terms)
        print(string)
    print()
