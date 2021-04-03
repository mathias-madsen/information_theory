import numpy as np
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


alphabet = "abcd"


if __name__ == "__main__":

    # assert grammar_is_normalized(codex)  # TODO: write normalizer
    grammar = Grammar(rulebooks=rulebooks, alphabet=alphabet)

    # plt.plot(expected_nonterminals(grammar), "o-")
    # plt.show()

    tree = grammar.sample_tree(root=0)
    sentence = tree.flatten()
    print("Sentence: %r\n" % sentence)
    print("Actual tree:\n")
    tree.pprint()
    print("log(probability) = %.5f.\n" % grammar.logprob(tree))

    # parse the sentence:
    inside = grammar.compute_inside_probabilities(sentence)
    assert inside[0, len(sentence) - 1, 0] > 0  # P(root = N_0) = 1
    initial = np.array([1.] + (len(grammar) - 1)*[0.])
    outside = grammar.compute_outside_probabilities(inside, initial=initial)
    # Compute the node-specific occupancy probabilities:
    posteriors = inside * outside
    posteriors /= posteriors[0, -1, :].sum()
    num_nonterminals_in_tree = 2*len(sentence) - 1
    assert np.isclose(posteriors[0, -1].sum(), 1.0)
    assert np.isclose(posteriors.sum(), num_nonterminals_in_tree)
    transprobs = grammar.sum_transition_probabilities(inside, outside)
    emitsprobs = grammar.sum_emission_probabilities(sentence, outside)

    most_probable_nodes = np.argmax(posteriors, axis=2)
    rows, cols = np.where(np.max(posteriors, axis=2) == 0)
    most_probable_nodes[rows, cols] = -1
    print("Actual node matrix:\n%s\n" % tree.nodematrix())
    print("Most probable nodes:\n%s\n" % most_probable_nodes)
    print("Actual occupancy matrix:\n%s\n" % (tree.nodematrix() != -1).astype(float))
    print("Occupancy probabilitie:\n%s\n" % posteriors.sum(axis=2).round(2))

    print("Actually occurred transitions:")
    for (k, i, j), count in tree.transition_counts().items():
        print("%s --> (%s, %s): %s" % (k, i, j, count))
    print()

    print("Transition probabilities:")
    for (k, i, j) in zip(*np.where(transprobs)):
        print("%s --> (%s, %s): %s" % (k, i, j, transprobs[k, i, j]))
    print()

    print("Actually occurred emissions:")
    for ((i, c), count) in tree.emission_counts().items():
        print("%s --> %r: %s" % (i, c, count))
    print("")

    print("Emission probabilities:")
    for idx, character in enumerate(grammar.alphabet):
        post = emitsprobs[:, idx]
        for k, prob in enumerate(post):
            if prob > 0:
                print("%s --> %r: %s" % (k, character, prob))
    print()

    N, S = grammar.emissions.shape
    randtrans = 0.2 * np.random.gamma(1.0, size=(N, N, N))
    randemits = 0.8 * np.random.gamma(1.0, size=(N, S))
    norms = np.sum(randtrans, axis=(1, 2)) + np.sum(randemits, axis=1)
    randtrans /= norms[:, None, None]
    randemits /= norms[:, None]

    estimated_grammar = Grammar(transitions=randtrans,
                                emissions=randemits,
                                alphabet=alphabet)

    trans_acc = np.zeros_like(grammar.transitions)
    emits_acc = np.zeros_like(grammar.emissions)

    trees = [grammar.sample_tree() for _ in range(100)]
    trees = [tree for tree in trees if tree.size <= 40]

    for tree in trees:
        sentence = tree.terminals
        print(".", end=" ", flush=True)
        # true_tp = np.zeros(3 * [len(grammar)])
        # for (k, i, j), p in tree.transition_counts().items():
        #     true_tp[k, i, j] += p
        # true_ep = np.zeros([len(grammar), 128])
        # for (k, c), p in  tree.emission_counts().items():
        #     true_ep[k, ord(c)] += p

        inside = grammar.compute_inside_probabilities(sentence)
        outside = grammar.compute_outside_probabilities(inside, initial=0)
        trans_acc += grammar.sum_transition_probabilities(inside, outside)
        emits_acc += grammar.sum_emission_probabilities(sentence, outside)
        # inside = estimated_grammar.compute_inside_probabilities(sentence)
        # outside = estimated_grammar.compute_outside_probabilities(inside, initial=0)
        # trans_acc += estimated_grammar.sum_transition_probabilities(inside, outside)
        # emits_acc += estimated_grammar.sum_emission_probabilities(sentence, outside)

    print("\n")

    tsums = np.sum(trans_acc, axis=(1, 2))
    esums = np.sum(emits_acc, axis=1)
    norms = tsums + esums
    new_transitions = trans_acc / norms[:, None, None]
    new_emissions = emits_acc / norms[:, None]

    print(grammar.transitions)
    print()
    print(new_transitions.round(2))
    print()
    print(grammar.transitions.sum(axis=(1, 2)))
    print(new_transitions.sum(axis=(1, 2)).round(2))
    print()
    print(grammar.emissions)
    print()
    print(new_emissions.round(2))
    print()
    print(grammar.emissions.sum(axis=(1,)))
    print(new_emissions.sum(axis=(1,)).round(2))
    print()

    print(sentence, "\n")
    print("Actual tree:")
    tree.pprint()
    print("Most likely tree:")
    grammar.compute_most_likely_tree(sentence).pprint()
    print("Some sampled trees:")
    grammar.conditionally_sample_tree(sentence).pprint()
    grammar.conditionally_sample_tree(sentence).pprint()
    grammar.conditionally_sample_tree(sentence).pprint()
