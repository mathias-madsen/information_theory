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

    for updatestep in range(50):

        print("--- Update %s ---\n" % (updatestep + 1,))

        # initialize counts with virtual observations:
        sumemits = 0.1 * model.emissions.copy()
        sumtrans = 0.1 * model.transitions.copy()

        print("Training . . .")
        true_likelihoods = []
        model_likelihoods = []
        for _ in tqdm(range(400), leave=False, unit="words"):
            actual_tree = reality.sample_tree(root=0)
            sentence = actual_tree.terminals
            if len(sentence) > 20:
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
        for _ in tqdm(range(200), leave=False, unit="samples"):
            sampled = model.sample_tree().terminals
            inner = reality.compute_inside_probabilities(sampled)
            was_impossible =  np.isclose(inner[0, -1, 0], 0.0)
            judgments.append(was_impossible)
        print()

        print("Average negative log-likelihood:")
        print("--------------------------------")
        print("Oracle: %.5f" % np.mean(-np.log(true_likelihoods)))
        print("Model: %.5f" % np.mean(-np.log(model_likelihoods)))
        print()

        print("Number of ungrammatical samples:")
        print("--------------------------------")
        print("%s / %s" % (sum(judgments), len(judgments)))
        print()

        norms = np.sum(sumemits, axis=1) + np.sum(sumtrans, axis=(1, 2))
        sumemits /= norms[:, None]
        sumtrans /= norms[:, None, None]

        model.update_from_matrices(sumtrans, sumemits)

        print()
