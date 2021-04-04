"""
The inside-outside algorithm for stochastic grammars.

Code for computing the conditional probability, given a stochastic
grammar, that a particular part of a sentence is the result of the
expansion of a particular nonterminal symbol.
"""

import numpy as np

from language_models.tree import Tree


class Grammar(dict):
    """
    Data structure that holds a stochastic, context-free grammar.
    """

    def __init__(self, rulebooks=None, alphabet=None, transitions=None, emissions=None):

        self.transitions = None
        self.emissions = None
        self.alphabet = None

        if rulebooks is not None:
            self.update_from_rulebooks(rulebooks, alphabet)
        elif transitions is not None and emissions is not None:
            self.update_from_matrices(transitions, emissions, alphabet)
        else:
            raise ValueError("Please provide either rulebooks or matrices.")
    
    def validate(self):
        """ Check that all stochastic constraints are satisfied. """

        # This check probably shouldn't live here, but we want to
        # verify that all the rulebooks are in Chomsky normal form:
        for nonterminal, rulebook in self.items():
            for expansion in rulebook.keys():
                branching = type(expansion) == tuple and len(expansion) == 2
                closing = type(expansion) in [str, int]
                assert branching or closing, (nonterminal, expansion)

        for nonterminal, rulebook in self.items():
            # note: `sum` can take iterables, `np.sum` cannot
            probsum = sum(rulebook.values())
            assert np.allclose(probsum, 1.0), (nonterminal, probsum)

        tsums = self.transitions.sum(axis=(1, 2))
        esums = self.emissions.sum(axis=1)
        assert np.allclose(tsums + esums, 1.0), (tsums + esums)
    
    def collect_terminals(self):
        """ Compile a set of terminals mentioned in the rulebooks. """

        terminals = set()

        for rulebook in self.values():
            for expansion in rulebook.keys():
                if type(expansion) == str:
                    terminals.add(expansion)
        
        return terminals
    
    def update_from_rulebooks(self, rulebooks, alphabet=None):
        """ Convert a dict of nonterminal rulebooks to matrix form. """

        self.update(rulebooks)

        if alphabet is not None:
            self.alphabet = tuple(alphabet)
        else:
            self.alphabet = tuple(sorted(self.collect_terminals()))

        num_nonterminals = len(rulebooks)
        transitions = np.zeros(3 * [num_nonterminals])
        emissions = np.zeros([num_nonterminals, len(self.alphabet)])

        for k, rulebook in self.items():
            for rule, probability in rulebook.items():
                if type(rule) == str:
                    # if we used dicts here, we could allow the
                    # alphabet to be infinite, but at the cost
                    # of having for parameterize the fallback
                    # distribution.
                    idx = self.alphabet.index(rule)
                    emissions[k, idx] += probability
                else:
                    i, j = rule
                    transitions[k, i, j] += probability

        self.transitions = transitions
        self.emissions = emissions
        self.validate()
    
    def update_from_matrices(self, transitions, emissions, alphabet=None):
        """ Convert transition and emission matrices to rulebooks. """

        self.transitions = transitions
        self.emissions = emissions

        if alphabet is not None:
            self.alphabet = tuple(alphabet)
        else:
            len_alphabet = self.emissions.shape[1]
            self.alphabet = tuple(chr(i) for i in range(len_alphabet))

        rulebooks = dict()

        for k, probij in enumerate(transitions):
            rulebooks[k] = dict()
            for i, probj in enumerate(probij):
                for j, prob in enumerate(probj):
                    rulebooks[k][i, j] = prob

        for k, probc in enumerate(emissions):
            for idx, prob in enumerate(probc):
                character = self.alphabet[idx]
                rulebooks[k][character] = prob
        
        self.update(rulebooks)
        self.validate()

    def sample_tree(self, root=0):
        """ Sample a random tree below a given nonterminal `root`. """

        distribution = self[root]
        expansions = list(distribution.keys())
        probabilities = [distribution[e] for e in expansions]
        idx = np.random.choice(len(expansions), p=probabilities)
        children = expansions[idx]
        if type(children) == tuple:
            return Tree(root, [self.sample_tree(nt) for nt in children])
        else:
            return Tree(root, [children])  # just a single terminal
    
    def get_root_distribution(self, root):
        """ Apply the conventional translations of an initial condition. """

        if root is None:  # no root condition given
            return np.ones(len(self)) / len(self)
        elif type(root) == int or np.isscalar(root):  # deterministic root
            assert 0 <= root < len(self)
            initial = np.zeros(len(self))
            initial[root] = 1.0
            return initial
        elif type(root) in [np.ndarray, list, tuple]:  # stochastic root
            assert np.shape(root) == (len(self),)
            return np.array(root)
        else:
            raise ValueError("Unpexpected root condition: %r" % root)

    def conditionally_sample_tree(self, sentence, inside=None, root=None):
        """ Sample a tree from the posterior distribution given a sentence. """
        
        if inside is None:
            inside = self.compute_inside_probabilities(sentence)
        
        dist = self.get_root_distribution(root)
        assert dist.shape == (len(self),)
        dist *= inside[0, -1, :]
        assert np.any(dist > 0)
        dist /= np.sum(dist)
        root = np.random.choice(len(self), p=dist)
        branchprobs = self.transitions[root, :, :]

        if inside.shape[0] == 1:
            assert len(sentence) == 1
            return Tree(root, sentence)

        assert np.any(branchprobs > 0)
        T = len(sentence)
        N = inside.shape[2]
        posteriors = np.zeros((T - 1, N, N))
        for cut in range(1, len(sentence)):
            leftprob = inside[0, cut - 1, :]
            rightprob = inside[cut, len(sentence) - 1, :]
            likelihoods = leftprob[:, None] * rightprob[None, :]
            posteriors[cut - 1, :, :] = branchprobs * likelihoods
        
        cutprobs = np.sum(posteriors, axis=(1, 2))
        assert np.any(cutprobs > 0)
        normed = cutprobs / np.sum(cutprobs)
        cut = 1 + np.random.choice(range(len(cutprobs)), p=normed)
        branchprob = posteriors[cut - 1, :, :]
        flatprob = np.reshape(branchprob, [-1]) / np.sum(branchprob)
        pair_idx = np.random.choice(branchprob.size, p=flatprob)
        root_0, root_1 = np.divmod(pair_idx, branchprob.shape[0])

        sentence_0 = sentence[:cut]
        inside_0 = inside[:cut, :cut, :]
        branch_0 = self.conditionally_sample_tree(sentence_0, inside_0, root_0)

        sentence_1 = sentence[cut:]
        inside_1 = inside[cut:, cut:, :]
        branch_1 = self.conditionally_sample_tree(sentence_1, inside_1, root_1)

        assert root_0 == branch_0.head
        assert root_1 == branch_1.head
        assert (root_0, root_1) in self[root]
        assert self[root][root_0, root_1] > 0

        return Tree(root, [branch_0, branch_1])
    
    def compute_most_likely_tree(self, sentence, inside=None, root=None):
        """ Find the most probable tree given the sentence. """

        inside = self.compute_maximal_probabilities(sentence)
        
        if np.isscalar(root):
            assert inside[0, -1, root] > 0
        dist = self.get_root_distribution(root)
        assert dist.shape == (len(self),)
        dist *= inside[0, -1, :]  # take the sentence likelihood into account
        assert np.any(dist > 0)
        dist /= np.sum(dist)
        root = np.argmax(dist)
        # print(dist, sentence, root)
        # print("P(%r) <= %s" % (sentence, dist[root]))

        if inside.shape[0] == 1:
            assert len(sentence) == 1
            return Tree(root, sentence)

        # from now on, condition on having to split `root`:
        branchprobs = self.transitions[root, :, :]
        assert np.any(branchprobs > 0)
        branchprobs /= np.sum(branchprobs)

        T, T, N = inside.shape
        joints = np.zeros((T - 1, N, N))
        for cut in range(1, T):
            leftprob = inside[0, cut - 1, :]
            rightprob = inside[cut, T - 1, :]
            likelihoods = leftprob[:, None] * rightprob[None, :]
            joints[cut - 1, :, :] = branchprobs * likelihoods
        raveled = np.argmax(joints)
        subcut, root_0, root_1 = np.unravel_index(raveled, joints.shape)
        cut = 1 + subcut
        branchprob = joints[cut - 1, :, :]
        assert np.sum(branchprob) > 0
        assert np.all(branchprob.max() == joints.max())

        assert joints[cut - 1, root_0, root_1] > 0

        sentence_0 = sentence[:cut]
        inside_0 = inside[:cut, :cut, :]
        branch_0 = self.compute_most_likely_tree(sentence_0, inside_0, root_0)
        assert self.logprob(branch_0) > -np.inf

        sentence_1 = sentence[cut:]
        inside_1 = inside[cut:, cut:, :]
        branch_1 = self.compute_most_likely_tree(sentence_1, inside_1, root_1)
        assert self.logprob(branch_1) > -np.inf

        assert sentence == sentence_0 + sentence_1
        assert root_0 == branch_0.head
        assert root_1 == branch_1.head
        assert (root_0, root_1) in self[root]
        assert self[root][root_0, root_1] > 0

        return Tree(root, [branch_0, branch_1])
    
    def logprob(self, tree):
        """ Compute the logarithmic probability of a tree in this grammar. """

        if len(tree) == 1:
            child = tree[0]  # a string terminal
            assert type(child) == str
            return np.log(self[tree.head][child])
        else:
            children = tuple(branch.head for branch in tree)
            logprob = np.log(self[tree.head][children])
            logprob += sum(self.logprob(branch) for branch in tree)
            return logprob
    
    def conditional_prob(self, tree, inner=None, root=None):
        """ Compute the conditional probability of tree given terminals. """

        if inner is None:
            inner = self.compute_inside_probabilities(tree.terminals)

        dist = self.get_root_distribution(root)
        denom = np.sum(dist * inner[0, -1, :])  # shapes (N,) --> ()
        
        if denom == 0.0:
            message = "Zero-probability condition: %r" % tree.terminals
            raise ZeroDivisionError(message)
        else:
            assert denom > 0, denom
        
        rootprob = dist[tree.head]

        if rootprob == 0.0:
            return 0.0

        lognumerator = self.logprob(tree) + np.log(rootprob)
        logconditional = lognumerator - np.log(denom)

        return np.exp(logconditional)

    def prob(self, tree):
        """ Compute the probability of a tree in this grammar. """

        return np.exp(self.logprob(tree))

    def compute_expected_lengths(self):
        """ Compute the mean num terminals under each nonterminal. """

        matrix = np.zeros((len(self) + 1, len(self) + 1))
        matrix[-1, -1] = 1.0

        for nonterminal, rulebook in self.items():
            for expansion, probability in rulebook.items():
                # case one: the rule expands the LHS into nonterminals:
                if type(expansion) == tuple:
                    for nt in expansion:
                        matrix[nonterminal, nt] += probability
                # case two: the rule expands the LHS into a terminal:
                else:
                    matrix[nonterminal, -1] += probability

        values, vectors = np.linalg.eig(matrix)

        if np.any(values > 1.0):
            return float("inf")
        
        idx, = np.where(values == 1.0)

        if len(idx) > 1:
            raise ValueError("The grammar contains unreachable nonterminals.")

        expectations = vectors[:, idx.item()]
        expectations /= expectations[-1]  # normalize using terminal length

        return np.real(expectations)
    
    def compute_maximal_probabilities(self, sentence):
        """ For each substring, compute the likelihood of the best tree. """

        len_sentence = len(sentence)
        inside = np.zeros(2*[len_sentence] + [len(self)])

        for start, character in enumerate(sentence):
            for k, rulebook in self.items():
                for rule, prob in rulebook.items():
                    if type(rule) == str and rule == character:
                        inside[start, start, k] = prob

        # TODO: fix this so it computes the right thing
        for width in range(2, len(sentence) + 1):
            for start in range(0, len(sentence) - width + 1):
                stop = start + width
                # maximize over places to split the setence and
                # pairs of child nodes that explain each part:
                maxprobs = np.zeros(len(self))
                for split in range(start + 1, stop):
                    left = inside[start, split - 1, :]
                    right = inside[split, stop - 1, :]
                    fork = self.transitions * left[:, None] * right[None, :]
                    fork = np.max(fork, axis=(1, 2))
                    maxprobs = np.maximum(maxprobs, fork)
                    inside[start, stop - 1, :] = maxprobs
        
        return inside


    def compute_inside_probabilities(self, sentence):
        """ Compute the likelihood of each substring given each nonterminal. """

        len_sentence = len(sentence)
        inside = np.zeros(2*[len_sentence] + [len(self)])

        for start, character in enumerate(sentence):
            for k, rulebook in self.items():
                for rule, probability in rulebook.items():
                    if type(rule) == str and rule == character:
                        inside[start, start, k] += probability

        for width in range(2, len(sentence) + 1):
            for start in range(0, len(sentence) - width + 1):
                stop = start + width
                for split in range(start + 1, stop):
                    left = inside[start, split - 1, :]
                    right = inside[split, stop - 1, :]
                    coverprobs = left[:, None] * right[None, :]
                    joints = np.sum(self.transitions * coverprobs, axis=(1, 2))
                    inside[start, stop - 1, :] += joints
        
        return inside

    def compute_outside_probabilities(self, inside, initial=None):
        """ Given letter-generation likelihoods, compute node probs.
        
        The result is an array of shape `[T, T, N]`, where `T` is the
        length of the sentence and `N` is the number of nonterminals.

        The entry at `(s, t, n)` contains the probability that the
        grammar would generate the letters from `0` to `s` and from
        `t` to `T`, as well as generate a node of type `n` at the top
        of the subtree spanning the letters from `s` to `t`.
        """

        assert inside.shape[0] == inside.shape[1]
        assert inside.shape[2] == len(self)

        len_sentence, _, num_nonterminals = inside.shape
        outside = np.zeros(inside.shape)

        if initial is None:  # no root condition given
            outside[0, len_sentence - 1, :] = 1.0 / num_nonterminals
        elif type(initial) == int:  # deterministic root
            outside[0, len_sentence - 1, :] = 0
            outside[0, len_sentence - 1, initial] = 1
        elif type(initial) in [np.ndarray, list, tuple]:  # stochastic root
            outside[0, len_sentence - 1, :] = initial
        else:
            raise ValueError("Unpexpected initial condition: %r" % initial)

        for width in reversed(range(1, len_sentence)):
            for start in range(0, len_sentence - width + 1):
                stop = start + width
                # case 1 -- parent to the left: there is a parent nonterminal
                # which generated the current head as well as several letters
                # to the left of whatever is spanned by the current head.
                for left in range(0, start):
                    parentprobs = outside[left, stop - 1, :]
                    branchprobs = self.transitions.transpose([1, 2, 0]) @ parentprobs
                    superprob = inside[left, start - 1] @ branchprobs
                    outside[start, stop - 1, :] += superprob
                # case 2 -- parent to the right: there is a parent nonterminal
                # which generated the current head as well as several letters
                # to the right of whatever is spanned by the current head.
                for right in range(stop + 1, len_sentence + 1):
                    parentprobs = outside[start, right - 1, :]
                    branchprobs = self.transitions.transpose([1, 2, 0]) @ parentprobs
                    superprob = inside[stop, right - 1] @ branchprobs.T
                    outside[start, stop - 1, :] += superprob

        assert outside.shape == inside.shape

        return outside

    def sum_transition_probabilities(self, inside, outside):
        """ Sum up joint expansion probabilities given the sentence.
        
        This returns a matrix of shape `(N, N, N)`, where `N` is the
        number of nonterminal symbols in the grammar.
        
        The entry at (k, i, j) contains the probability that the
        nonterminal k was expanded into the pair of nonterminals (i, j)
        somewhere in the syntatic tree of this sentence _given_ that
        the sentence occurred.

        Since each terminal letter except one is explained by one
        branching transition, the sum of the probabilities in this
        table sum to the length of the sentence minus 1.
        """

        # probabilities that the root is a nonterminal of type `k`:
        rootprobs = inside[0, -1, :] * outside[0, -1, :]
        # probability that the sentence has _any_ root at all:
        probability_of_sentence = np.sum(rootprobs)
        assert probability_of_sentence > 0, probability_of_sentence
        # for each slot in the tree matrix, the conditional probability
        # that this slot is occupied by a given type of nonterminal,
        # given that the sentence occurred:
        nodeprobs = outside / probability_of_sentence

        len_sentence, _, num_nonterminals = inside.shape
        joints = np.zeros(3 * [num_nonterminals])
        for start in range(len_sentence):
            for stop in range(start + 2, len_sentence + 1):
                priors = nodeprobs[start, stop - 1, :]
                likelihood = np.zeros_like(self.transitions)
                for split in range(start + 1, stop):
                    qi = inside[start, split - 1, :][None, :, None]
                    qj = inside[split, stop - 1, :][None, None, :]
                    likelihood += qi * qj * self.transitions
                joints += priors[:, None, None] * likelihood

        len_sentence = inside.shape[0]
        assert np.isclose(joints.sum(), len_sentence - 1), joints.sum()

        return joints
    
    def sum_emission_probabilities(self, sentence, outside):

        _, _, num_nonterminals = outside.shape
        probs = np.zeros((num_nonterminals, len(self.alphabet)))

        for start, character in enumerate(sentence):
            idx = self.alphabet.index(character)
            priors = outside[start, start, :]  # shape (N,)
            likelihoods = self.emissions[:, idx]  # shape (N,)
            posteriors = priors * likelihoods
            if np.any(posteriors > 0):
                posteriors /= posteriors.sum()
            probs[:, idx] += posteriors
        
        return probs
