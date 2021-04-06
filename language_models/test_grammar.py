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


def test_that_grammar_from_rulebooks_compiles_alphabet_correctly():

    rulebooks = {0: {"a": 0.5, "d": 0.5}, 1: {"c": 0.3, "b": 0.7}}

    grammar = Grammar(rulebooks)
    assert grammar.alphabet == tuple("abcd")

    grammar = Grammar(rulebooks, alphabet=None)  # same as above
    assert grammar.alphabet == tuple("abcd")

    grammar = Grammar(rulebooks, alphabet="abcdX")
    assert grammar.alphabet == tuple("abcdX")

    grammar = Grammar(rulebooks, alphabet=list("abcdX"))
    assert grammar.alphabet == tuple("abcdX")


def test_that_grammar_equality_works_as_expected():

    rulebooks1 = rulebooks.copy()
    
    rulebooks2 = rulebooks.copy()
    rulebooks2[3] = {"c": 0.3, "d": 0.7}
    
    rulebooks3 = rulebooks.copy()
    rulebooks3[0] = {(1, 1): 1.0}

    grammar0 = Grammar(rulebooks)
    grammar1 = Grammar(rulebooks1)
    grammar2 = Grammar(rulebooks2)
    grammar3 = Grammar(rulebooks3)

    assert grammar0 == grammar0
    assert grammar0 == grammar1
    assert grammar1 == grammar1

    assert grammar2 == grammar2
    assert grammar0 != grammar2
    assert grammar2 != grammar0

    assert grammar0 != grammar3
    assert grammar1 != grammar3
    assert grammar2 != grammar3


def test_that_grammar_computes_probabilities_in_the_right_range():

    grammar = Grammar(rulebooks)

    for tree in grammar.sample_tree():
        logprob = grammar.logprob_tree(tree)
        prob = grammar.prob_tree(tree)
        assert -np.inf < logprob <= 0.0
        assert 0.0 < prob <= 1.0
        assert np.isclose(np.exp(logprob), prob)


def test_that_tree_probs_agree_with_explicit_computations():

    # define the grammar here to make sure it stays put:

    frozen = {
        0: {(1, 1): 0.4, (1, 2): 0.6},
        1: {(1, 2): 0.2, (2, 1): 0.3, 'a': 0.5},
        2: {(1, 3): 0.1, (1, 2): 0.2, 'b': 0.7},
        3: {'c': 0.6, 'd': 0.4}
        }

    grammar = Grammar(frozen)

    tree1 = Tree(1, ["a"])
    prob1 = grammar.prob_tree(tree1)
    assert np.isclose(prob1, 0.5)

    tree2 = Tree(2, ["b"])
    prob2 = grammar.prob_tree(tree2)
    assert np.isclose(prob2, 0.7)

    tree3 = Tree(1, [tree1, tree2])  # 1 --> (1, 2)
    prob3 = grammar.prob_tree(tree3)
    assert np.isclose(prob3, 0.2 * prob1 * prob2)

    tree4 = Tree(0, [tree1, tree3])  # 0 --> (1, 1)
    prob4 = grammar.prob_tree(tree4)
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
    best_logprob = grammar.logprob_tree(best_tree)

    for _ in range(20):
        tree = grammar.compute_most_likely_tree(sentence, inner, root=0)
        assert np.isclose(grammar.logprob_tree(tree), best_logprob)


def test_that_most_probable_tree_is_most_probable():

    grammar = Grammar(rulebooks)

    actual_tree = grammar.sample_tree(root=0)
    actual_logprob = grammar.logprob_tree(actual_tree)
    sentence = actual_tree.terminals

    inner = grammar.compute_inside_probabilities(sentence)
    max_logprob = np.sum(inner[0, -1, :])
    best_tree = grammar.compute_most_likely_tree(sentence, root=0)
    best_logprob = grammar.logprob_tree(best_tree)
    assert actual_logprob <= best_logprob + 1e-14, (actual_logprob, best_logprob)
    assert best_logprob <= max_logprob  # total likelihood across trees

    for _ in range(10):
        random_tree = grammar.conditionally_sample_tree(sentence, inner, root=0)
        random_logprob = grammar.logprob_tree(random_tree)
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
    conditional_logprob = grammar.logprob_tree(tree) - np.log(word_likelihood)
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


def test_that_emission_probabilities_are_in_the_right_range():

    # first define a grammar with some overlap between the
    # terminals that can be produced by different nonterminals:
    rulebooks = {
        0: {(1, 1): 0.3, (1, 2): 0.4, 'a': 0.3},
        1: {(1, 2): 0.2, (2, 1): 0.3, 'b': 0.4, 'a': 0.1},
        2: {'a': 0.3, 'b': 0.7}
        }

    grammar = Grammar(rulebooks)
    tree = grammar.sample_tree(root=0)
    sentence = tree.terminals

    inside = grammar.compute_inside_probabilities(sentence)
    outside = grammar.compute_outside_probabilities(inside, initial=0)
    probsums = grammar.sum_emission_probabilities(sentence, outside)

    assert np.all(probsums >= 0.0)  # but may be larger than 1.0

    for (nt, letter) in tree.iter_emissions():
        idx = grammar.alphabet.index(letter)
        assert probsums[nt, idx] > 0.0  # must be positive if it happened


def test_that_emission_probabilities_are_well_calibrated():

    # first define a grammar with some overlap between the
    # terminals that can be produced by different nonterminals:
    rulebooks = {
        0: {(1, 1): 0.3, (1, 2): 0.4, 'a': 0.3},
        1: {(1, 2): 0.2, (2, 1): 0.3, 'b': 0.4, 'a': 0.1},
        2: {'a': 0.3, 'b': 0.7}
        }

    grammar = Grammar(rulebooks)
    tree = grammar.sample_tree(root=0)
    sentence = tree.terminals

    inside = grammar.compute_inside_probabilities(sentence)
    outside = grammar.compute_outside_probabilities(inside, initial=0)
    probsums = grammar.sum_emission_probabilities(sentence, outside)

    num_samples = 400
    occursums = np.zeros_like(probsums)
    for _ in range(num_samples):
        randtree = grammar.conditionally_sample_tree(sentence, inside, root=0)
        for (nt, letter) in randtree.iter_emissions():
            idx = grammar.alphabet.index(letter)
            occursums[nt, idx] += 1
    occursums /= num_samples

    assert np.allclose(probsums, occursums, atol=0.1)


def test_that_transition_probabilities_are_in_the_right_range():

    grammar = Grammar(rulebooks)
    tree = grammar.sample_tree(root=0)
    sentence = tree.terminals

    inside = grammar.compute_inside_probabilities(sentence)
    outside = grammar.compute_outside_probabilities(inside, initial=0)
    probsums = grammar.sum_transition_probabilities(inside, outside)

    assert np.all(probsums >= 0.0)  # but may be larger than 1.0

    for triple in tree.iter_transitions():
        assert probsums[triple] > 0.0


def test_that_transition_probabilities_are_well_calibrated():

    grammar = Grammar(rulebooks)
    tree = grammar.sample_tree(root=0)
    sentence = tree.terminals

    inside = grammar.compute_inside_probabilities(sentence)
    outside = grammar.compute_outside_probabilities(inside, initial=0)
    probsums = grammar.sum_transition_probabilities(inside, outside)

    num_samples = 400
    occursums = np.zeros_like(probsums)
    for _ in range(num_samples):
        randtree = grammar.conditionally_sample_tree(sentence, inside, root=0)
        for triple in randtree.iter_transitions():
            occursums[triple] += 1
    occursums /= num_samples

    assert np.allclose(probsums, occursums, atol=0.1)


def test_that_parameter_estimation_is_consistent():

    grammar = Grammar(rulebooks)

    sumemits = np.zeros_like(grammar.emissions)
    sumtrans = np.zeros_like(grammar.transitions)

    for _ in range(400):
        tree = grammar.sample_tree(root=0)
        word = tree.terminals
        if len(word) > 25:
            continue  # introduces slight bias, but saves a ton of time
        inside = grammar.compute_inside_probabilities(word)
        outside = grammar.compute_outside_probabilities(inside, initial=0)
        sumemits += grammar.sum_emission_probabilities(word, outside)
        sumtrans += grammar.sum_transition_probabilities(inside, outside)
    
    norms = np.sum(sumemits, axis=1) + np.sum(sumtrans, axis=(1, 2))
    sumemits /= norms[:, None]
    sumtrans /= norms[:, None, None]

    assert np.allclose(grammar.emissions, sumemits, atol=0.1)
    assert np.allclose(grammar.transitions, sumtrans, atol=0.1)


if __name__ == "__main__":

    test_that_grammar_from_rulebooks_compiles_alphabet_correctly()
    test_that_grammar_equality_works_as_expected()
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
    test_that_emission_probabilities_are_in_the_right_range()
    test_that_emission_probabilities_are_well_calibrated()
    test_that_transition_probabilities_are_in_the_right_range()
    test_that_transition_probabilities_are_well_calibrated()
    test_that_parameter_estimation_is_consistent()
