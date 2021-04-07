"""
Demo script showing that the learning of a stocastic, context-free
grammar using the EM algorithm in the well-specified case.

The target grammar assigns probability zero to a large subset of all
possible sentences, so we can evaluate the estimated grammar on its
rate of false positives (the probability of sampling a sentence that
the true target grammar considers impossible).

Training takes on the order of 100 update steps, with each update step
taking on the order of 10 seconds (on my machine).
"""

import numpy as np
from tqdm import tqdm

from language_models.grammar import Grammar


rulebooks = {

0: {
    (1, 1): 0.4,
    (1, 2): 0.6,
    },

1: {
    (1, 2): 0.2,
    (2, 1): 0.3,
    "a": 0.5,
    },

2: {
    (1, 3): 0.1,
    (1, 2): 0.2,
    "b": 0.7,
    },

3: {
    "c": 0.6,
    "d": 0.4,
    },

}


def sample_random_grammar_like(grammar):

    transitions = np.random.gamma(1.0, size=grammar.transitions.shape)
    emissions = np.random.gamma(1.0, size=grammar.emissions.shape)

    transitions /= np.sum(transitions, axis=(1, 2), keepdims=True)
    emissions /= np.sum(emissions, axis=1, keepdims=True)

    transitions *= 0.3
    emissions *= 0.7

    return Grammar(transitions=transitions,
                   emissions=emissions,
                   alphabet=grammar.alphabet)


if __name__ == "__main__":

    reality = Grammar(rulebooks)
    model = sample_random_grammar_like(reality)

    for updatestep in range(100):

        print("--- Update %s ---\n" % (updatestep + 1,))

        # initialize counts with virtual observations:
        sumemits = 10.0 * model.emissions.copy()
        sumtrans = 10.0 * model.transitions.copy()

        print("Training . . .")
        true_likelihoods = []
        model_likelihoods = []
        for _ in tqdm(range(1000), leave=False, unit="words"):
            actual_tree = reality.sample_tree(root=0)
            sentence = actual_tree.terminals
            if len(sentence) > 30 and updatestep <= 25:
                continue  # slight bias, massive speedup
            inside = model.compute_inside_probabilities(sentence)
            outside = model.compute_outside_probabilities(inside, initial=0)
            sumemits += model.sum_emission_probabilities(sentence, outside)
            sumtrans += model.sum_transition_probabilities(inside, outside)
            true_inner = reality.compute_inside_probabilities(sentence)
            true_likelihoods.append(true_inner[0, -1, 0])
            model_likelihoods.append(inside[0, -1, 0])
        print()

        print("Validating . . .")
        judgments = []
        for _ in tqdm(range(300), leave=False, unit="samples"):
            sampled = model.sample_tree().terminals
            inner = reality.compute_inside_probabilities(sampled)
            judgments.append(inner[0, -1, 0] == 0.0)
        print()

        print("Average negative log-likelihood:")
        print("--------------------------------")
        print("Oracle: %.5f" % np.mean(-np.log(true_likelihoods)))
        print("Model: %.5f" % np.mean(-np.log(model_likelihoods)))
        print("False positives: %s / %s" % (sum(judgments), len(judgments)))
        print()

        norms = np.sum(sumemits, axis=1) + np.sum(sumtrans, axis=(1, 2))
        sumemits /= norms[:, None]
        sumtrans /= norms[:, None, None]

        model.update_from_matrices(sumtrans, sumemits)

        print()
