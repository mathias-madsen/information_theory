import numpy as np
from language_models.grammar import Grammar
from language_models.tree import Tree


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


def test_that_grammar_from_rulebooks_compiles_alphabet_correctly():

    grammar = Grammar(rulebooks)
    assert grammar.alphabet == tuple("abcd")

    grammar = Grammar(rulebooks, alphabet="abcdX")
    assert grammar.alphabet == tuple("abcdX")

    grammar = Grammar(rulebooks, alphabet=list("abcdX"))
    assert grammar.alphabet == tuple("abcdX")


def test_that_grammar_computes_probabilities_in_the_right_range():

    grammar = Grammar(rulebooks)

    for tree in grammar.sample_tree():
        logprob = grammar.logprob(tree)
        prob = grammar.prob(tree)
        assert -np.inf < logprob <= 0.0
        assert 0.0 < prob <= 1.0
        assert np.isclose(np.exp(logprob), prob)


def test_that_tree_probs_agree_with_explicit_computations():

    grammar = Grammar(rulebooks)

    tree1 = Tree(1, ["a"])
    prob1 = grammar.prob(tree1)
    assert np.isclose(prob1, 0.5)

    tree2 = Tree(2, ["b"])
    prob2 = grammar.prob(tree2)
    assert np.isclose(prob2, 0.7)

    tree3 = Tree(1, [tree1, tree2])  # 1 --> (1, 2)
    prob3 = grammar.prob(tree3)
    assert np.isclose(prob3, 0.2 * prob1 * prob2)

    tree4 = Tree(0, [tree1, tree3])  # 0 --> (1, 1)
    prob4 = grammar.prob(tree4)
    assert np.isclose(prob4, 0.4 * prob1 * prob3)


def test_that_total_likelihood_exceeds_single_largest_likelihood():

    grammar = Grammar(rulebooks)
    tree = grammar.sample_tree()
    sentence = tree.terminals

    maximal = grammar.compute_maximal_probabilities(sentence)
    summed = grammar.compute_inside_probabilities(sentence)

    assert np.all(maximal <= summed + 1e-12), (sentence, maximal, summed)


def pairwise_equalities(things):
    """ Return an array of item comparisons, avoiding self-comparisons. """

    return np.array([things[i] == things[j] for i in range(len(things))
                     for j in range(i + 1, len(things))])


def test_that_sample_returns_trees_with_different_terminals():

    grammar = Grammar(rulebooks)
    trees = [grammar.sample_tree(root=0) for _ in range(100)]
    words = ["".join(tree.terminals) for tree in trees]
    
    assert len(set(words)) > 1


def test_that_conditional_sampling_returns_different_trees():

    grammar = Grammar(rulebooks)
    word = "abba"  # structually ambiguous under the grammar
    inner = grammar.compute_inside_probabilities(word)
    trees = [grammar.conditionally_sample_tree(word, inner, root=0)
             for _ in range(30)]
    
    # we sample different trees, but they all have the same leaves:
    assert set(["".join(tree.terminals) for tree in trees]) == set([word])

    comparisons = pairwise_equalities(trees)
    assert any(comparisons)  # it virtually impossible they're all identical
    assert not all(comparisons)  # they cannot all be different either


def test_that_most_probable_always_returns_a_tree_of_the_same_logprob():
    """
    Because of slight numerical instabilities, the most-likely-tree
    method does _not_ in fact always return the same tree when there
    are multiple trees with the same or very nearly the same likelihood.

    The best we can ask for is that the method returns _a_ best tree,
    defined as one with the biggest likelihood, or nearly the biggest.
    """

    grammar = Grammar(rulebooks)
    actual_tree = grammar.sample_tree()
    sentence = actual_tree.terminals
    inner = grammar.compute_inside_probabilities(sentence)
    best_tree = grammar.compute_most_likely_tree(sentence, inner, root=0)
    best_logprob = grammar.logprob(best_tree)

    for _ in range(20):
        tree = grammar.compute_most_likely_tree(sentence, inner, root=0)
        assert np.isclose(grammar.logprob(tree), best_logprob)


def test_that_most_probable_tree_is_most_probable():

    grammar = Grammar(rulebooks)

    actual_tree = grammar.sample_tree(root=0)
    actual_logprob = grammar.logprob(actual_tree)
    sentence = actual_tree.terminals

    inner = grammar.compute_inside_probabilities(sentence)
    max_logprob = np.sum(inner[0, -1, :])
    best_tree = grammar.compute_most_likely_tree(sentence, root=0)
    best_logprob = grammar.logprob(best_tree)
    assert actual_logprob <= best_logprob + 1e-14, (actual_logprob, best_logprob)
    assert best_logprob <= max_logprob  # total likelihood across trees

    for _ in range(10):
        random_tree = grammar.conditionally_sample_tree(sentence, inner, root=0)
        random_logprob = grammar.logprob(random_tree)
        assert random_logprob <= best_logprob + 1e-14, (random_logprob, best_logprob)


def test_that_conditional_probability_agrees_with_alternative_computation():

    grammar = Grammar(rulebooks)
    tree = grammar.sample_tree(root=0)
    word = tree.terminals
    
    # compute the total likelihood of the letters, summed over all trees:
    inner = grammar.compute_inside_probabilities(word)
    word_likelihood = inner[0, -1, 0]  # assume root head is 0
    assert word_likelihood > 0

    # compute the conditional probability of the tree given the letters:
    conditional_logprob = grammar.logprob(tree) - np.log(word_likelihood)
    conditional_prob = np.exp(conditional_logprob)

    # perform the same computation using the method:
    by_method = grammar.conditional_prob(tree, root=0)

    assert np.isclose(conditional_prob, by_method)


def test_that_tree_prob_is_well_calibrated():

    grammar = Grammar(rulebooks)

    while True:
        tree = grammar.sample_tree(root=0)
        word = tree.terminals
        if len(word) <= 10:  # it takes forever with long words
            break
    
    inner = grammar.compute_inside_probabilities(word)
    conditional_prob = grammar.conditional_prob(tree, inner, root=0)

    # empiricall estimate the conditional frequency of the tree:
    sample = lambda: grammar.conditionally_sample_tree(word, inner, root=0)
    num_samples = 400
    empirical = np.mean([tree == sample() for _ in range(num_samples)])

    # compute the standard deviation of the average, and cast a wide net:
    std = 0.25 / num_samples ** 0.5
    atol = 5.0 * std  # 5 stds ~ about one in a million chance

    # check that the empirical and theoretical frequencies roughly agree:
    assert np.isclose(empirical, conditional_prob, atol=atol)


def test_that_occupancy_probabilities_are_in_the_unit_interval():

    grammar = Grammar(rulebooks)
    tree = grammar.sample_tree(root=0)
    fillprob = grammar.compute_occupancy_probabilities(tree.terminals)

    assert np.all(fillprob >= 0.0 - 1e-12)
    assert np.all(fillprob <= 1.0 + 1e-12)


def test_that_occupancy_probabilities_are_zero_and_one_where_expected():

    grammar = Grammar(rulebooks)
    tree = grammar.sample_tree(root=0)
    
    fillprob = grammar.compute_occupancy_probabilities(tree.terminals)
    assert np.allclose(fillprob.diagonal(), 1.0)  # terminals have parents
    
    should_be_upper = fillprob - np.diag(fillprob.diagonal())
    assert np.allclose(np.tril(should_be_upper), 0)  # no p>0 below diagonal

    # each branching gives rise to one extra nonterminals and one
    # extra terminal; exactly `size` nonterminals are not going to
    # branch out. Together, this implies the following constraint:
    expected_probsum = 2*tree.size - 1
    computed_probsum = np.sum(fillprob)
    assert np.isclose(expected_probsum, computed_probsum)


def test_that_occupancy_probabilities_are_not_zero_at_filled_slots():

    grammar = Grammar(rulebooks)
    tree = grammar.sample_tree(root=0)
    isfilled = tree.get_occupancy_matrix(dtype=float)
    fillprob = grammar.compute_occupancy_probabilities(tree.terminals, root=0)

    assert np.all(isfilled[fillprob == 0.0] == 0)  # p = 0 ==> not filled
    assert np.all(fillprob[isfilled == 1.0] > 0)  # same: filled ==> p > 0


if __name__ == "__main__":

    test_that_grammar_from_rulebooks_compiles_alphabet_correctly()
    test_that_grammar_computes_probabilities_in_the_right_range()
    test_that_tree_probs_agree_with_explicit_computations()
    test_that_total_likelihood_exceeds_single_largest_likelihood()
    test_that_sample_returns_trees_with_different_terminals()
    test_that_conditional_sampling_returns_different_trees()
    test_that_most_probable_always_returns_a_tree_of_the_same_logprob()
    test_that_most_probable_tree_is_most_probable()
    test_that_conditional_probability_agrees_with_alternative_computation()
    test_that_tree_prob_is_well_calibrated()
    test_that_occupancy_probabilities_are_not_zero_at_filled_slots()
    
    # # assert grammar_is_normalized(codex)  # TODO: write normalizer
    # grammar = Grammar(rulebooks=rulebooks, alphabet=alphabet)

    # # plt.plot(expected_nonterminals(grammar), "o-")
    # # plt.show()

    # tree = grammar.sample_tree(root=0)
    # sentence = tree.flatten()
    # print("Sentence: %r\n" % sentence)
    # print("Actual tree:\n")
    # tree.pprint()
    # print("log(probability) = %.5f.\n" % grammar.logprob(tree))

    # # parse the sentence:
    # inside = grammar.compute_inside_probabilities(sentence)
    # assert inside[0, len(sentence) - 1, 0] > 0  # P(root = N_0) = 1
    # initial = np.array([1.] + (len(grammar) - 1)*[0.])
    # outside = grammar.compute_outside_probabilities(inside, initial=initial)
    # # Compute the node-specific occupancy probabilities:
    # posteriors = inside * outside
    # posteriors /= posteriors[0, -1, :].sum()
    # num_nonterminals_in_tree = 2*len(sentence) - 1
    # assert np.isclose(posteriors[0, -1].sum(), 1.0)
    # assert np.isclose(posteriors.sum(), num_nonterminals_in_tree)
    # transprobs = grammar.sum_transition_probabilities(inside, outside)
    # emitsprobs = grammar.sum_emission_probabilities(sentence, outside)

    # most_probable_nodes = np.argmax(posteriors, axis=2)
    # rows, cols = np.where(np.max(posteriors, axis=2) == 0)
    # most_probable_nodes[rows, cols] = -1
    # print("Actual node matrix:\n%s\n" % tree.nodematrix())
    # print("Most probable nodes:\n%s\n" % most_probable_nodes)
    # print("Actual occupancy matrix:\n%s\n" % (tree.nodematrix() != -1).astype(float))
    # print("Occupancy probabilitie:\n%s\n" % posteriors.sum(axis=2).round(2))

    # print("Actually occurred transitions:")
    # for (k, i, j), count in tree.transition_counts().items():
    #     print("%s --> (%s, %s): %s" % (k, i, j, count))
    # print()

    # print("Transition probabilities:")
    # for (k, i, j) in zip(*np.where(transprobs)):
    #     print("%s --> (%s, %s): %s" % (k, i, j, transprobs[k, i, j]))
    # print()

    # print("Actually occurred emissions:")
    # for ((i, c), count) in tree.emission_counts().items():
    #     print("%s --> %r: %s" % (i, c, count))
    # print("")

    # print("Emission probabilities:")
    # for idx, character in enumerate(grammar.alphabet):
    #     post = emitsprobs[:, idx]
    #     for k, prob in enumerate(post):
    #         if prob > 0:
    #             print("%s --> %r: %s" % (k, character, prob))
    # print()

    # N, S = grammar.emissions.shape
    # randtrans = 0.2 * np.random.gamma(1.0, size=(N, N, N))
    # randemits = 0.8 * np.random.gamma(1.0, size=(N, S))
    # norms = np.sum(randtrans, axis=(1, 2)) + np.sum(randemits, axis=1)
    # randtrans /= norms[:, None, None]
    # randemits /= norms[:, None]

    # estimated_grammar = Grammar(transitions=randtrans,
    #                             emissions=randemits,
    #                             alphabet=alphabet)

    # trans_acc = np.zeros_like(grammar.transitions)
    # emits_acc = np.zeros_like(grammar.emissions)

    # trees = [grammar.sample_tree() for _ in range(1000)]
    # trees = [tree for tree in trees if tree.size <= 40]

    # for tree in trees:
    #     sentence = tree.terminals
    #     print(".", end=" ", flush=True)
    #     # true_tp = np.zeros(3 * [len(grammar)])
    #     # for (k, i, j), p in tree.transition_counts().items():
    #     #     true_tp[k, i, j] += p
    #     # true_ep = np.zeros([len(grammar), 128])
    #     # for (k, c), p in  tree.emission_counts().items():
    #     #     true_ep[k, ord(c)] += p

    #     inside = grammar.compute_inside_probabilities(sentence)
    #     outside = grammar.compute_outside_probabilities(inside, initial=0)
    #     trans_acc += grammar.sum_transition_probabilities(inside, outside)
    #     emits_acc += grammar.sum_emission_probabilities(sentence, outside)
    #     # inside = estimated_grammar.compute_inside_probabilities(sentence)
    #     # outside = estimated_grammar.compute_outside_probabilities(inside, initial=0)
    #     # trans_acc += estimated_grammar.sum_transition_probabilities(inside, outside)
    #     # emits_acc += estimated_grammar.sum_emission_probabilities(sentence, outside)

    # print("\n")

    # tsums = np.sum(trans_acc, axis=(1, 2))
    # esums = np.sum(emits_acc, axis=1)
    # norms = tsums + esums
    # new_transitions = trans_acc / norms[:, None, None]
    # new_emissions = emits_acc / norms[:, None]

    # print(grammar.transitions)
    # print()
    # print(new_transitions.round(2))
    # print()
    # print(grammar.transitions.sum(axis=(1, 2)))
    # print(new_transitions.sum(axis=(1, 2)).round(2))
    # print()
    # print(grammar.emissions)
    # print()
    # print(new_emissions.round(2))
    # print()
    # print(grammar.emissions.sum(axis=(1,)))
    # print(new_emissions.sum(axis=(1,)).round(2))
    # print()

    # print(sentence, "\n")
    # print("Actual tree:")
    # tree.pprint()
    # print("Most likely tree:")
    # grammar.compute_most_likely_tree(sentence).pprint()
    # print("Some sampled trees:")
    # grammar.conditionally_sample_tree(sentence).pprint()
    # grammar.conditionally_sample_tree(sentence).pprint()
    # grammar.conditionally_sample_tree(sentence).pprint()