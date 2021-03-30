"""
The inside-outside algorithm for stochastic grammars.

Code for computing the conditional probability, given a stochastic
grammar, that a particular part of a sentence is the result of the
expansion of a particular nonterminal symbol.
"""

from collections import defaultdict
from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np


class Tree(list):

    def __init__(self, head, elms):
        """ Create a tree with the name `head` and children `elms`. """
        list.__init__(self, elms)
        self.head = head
        if len(elms) == 0:
            raise ValueError("Trivial leaves prohibited: %r" % (elms,))
        if len(elms) == 1 and type(elms[0]) == Tree:
            raise ValueError("Trivial subtrees prohibited: %r" % (elms,))
        self.terminals = self.collect_terminals()
        self.size = len(self.terminals)

    def __repr__(self):
        """ Produce the call that would produce this tree. """
        return "Tree(%r, %r)" % (self.head, list(self))

    def collect_terminals(self):
        """ Flatten the tree into a list of terminal strings. """
        return sum([b.terminals if type(b) == Tree else [b] for b in self], [])

    def flatten(self, sep=""):
        """ Convert the tree into a sentence. """
        return sep.join(str(term) for term in self.terminals)

    def pprint(self, depth=0, indent="  "):
        """ Pretty-print the tree as a nested structure. """
        margin = depth * indent
        if len(self) == 1:
            print(margin + "%r: %r" % (self.head, self[0]))
        else:
            print(margin + "%r:" % (self.head,))
            for branch in self:
                branch.pprint(depth=depth + 1, indent=indent)
        if depth == 0:
            print("")  # empty line at the very end

    def spandict(self):
        """ Convert the tree into a dict of head spans. """
        size = self.size
        spans = {(0, size): self.head}
        if len(self) == 1:
            return spans
        left, right = self
        shift = left.size
        spans.update(left.spandict())
        spans.update({(shift + i, shift + j): head
                      for (i, j), head in right.spandict().items()})
        return spans
    
    def nodematrix(self):
        """ Convert the tree into a matrix of node names. """
        size = self.size
        spans = self.spandict()
        matrix = [[-1 for _ in range(size)] for _ in range(size)]
        for (i, j), head in spans.items():
            matrix[i][j - 1] = head
        return np.array(matrix)


def extract_most_likely_tree(singles, grammar, sentence, start=None, stop=None, root="N0"):
    """ Compute the most likely parse of a sentence. """
    if start is None:
        start, stop = 0, len(sentence)
    if stop == start + 1:
        return Tree(root, [sentence[start]])
    options = dict()
    for rule, ruleprob in grammar[root].items():
        if type(rule) == str:
            continue  # a terminal can't match >= 2 characters
        Nleft, Nright = rule
        for mid in range(start + 1, stop):
            Pleft = singles[start, mid][Nleft]
            Pright = singles[mid, stop][Nright]
            options[Nleft, Nright, mid] = ruleprob * Pleft * Pright
    assert any(options.values()), "Error: no parse of %s" % options
    Nleft, Nright, mid = max(options, key=lambda k: options[k])
    assert options[Nleft, Nright, mid] > 0
    Tleft = extract_most_likely_tree(singles, grammar, sentence, start, mid, Nleft)
    Tright = extract_most_likely_tree(singles, grammar, sentence, mid, stop, Nright)
    assert Nleft == Tleft.head
    assert Nright == Tright.head
    return Tree(root, [Tleft, Tright])


def conditionally_sample_tree(singles, grammar, sentence, start=None, stop=None, root="N0"):
    """ Sample a tree that consistent with the given sentence. """
    # in the base case, we cover the whole sentence:
    if start is None:
        start, stop = 0, len(sentence)
    if stop == start + 1:
        return Tree(root, [sentence[start]])
    # find most probable bifurcation of that node:
    splits = []
    splitprobs = []
    for rule, ruleprob in grammar[root].items():
        if type(rule) == str:
            continue  # a terminal can't match >= 2 characters
        Nleft, Nright = rule
        for mid in range(start + 1, stop):
            Pleft = singles[start, mid][Nleft]
            Pright = singles[mid, stop][Nright]
            splits.append((Nleft, Nright, mid))
            splitprobs.append(ruleprob * Pleft * Pright)
    splitprobs = np.array(splitprobs) / sum(splitprobs)
    assert any(splitprobs > 0), "Error: no possibilities in %s" % rule
    index = np.random.choice(len(splits), p=splitprobs)
    Nleft, Nright, mid = splits[index]
    Tleft = extract_most_likely_tree(singles, grammar, sentence, start, mid, Nleft)
    Tright = extract_most_likely_tree(singles, grammar, sentence, mid, stop, Nright)
    return Tree(root, [Tleft, Tright])


class Grammar(dict):

    def __init__(self, rulebooks):

        # Add the given rules to the internal library:
        self.update(rulebooks)

        # compile a transition matrix from the rules:
        self.transitions = np.zeros(3 * [len(self)])
        for k, rulebook in self.items():
            for rule, probability in rulebook.items():
                if type(rule) == str:
                    # at this point, we still allow the space of terminals to
                    # be infinite, so we don't compile a matrix of emission
                    # probabilities, but instead use the grammar rules directly
                    continue
                else:
                    i, j = rule
                    self.transitions[k, i, j] += probability

    def sample(self, root=0):
        """ Sample a random tree below a given nonterminal `root`. """

        distribution = self[root]
        expansions = list(distribution.keys())
        probabilities = [distribution[e] for e in expansions]
        idx = np.random.choice(len(expansions), p=probabilities)
        children = expansions[idx]
        if type(children) == tuple:
            return Tree(root, [self.sample(nt) for nt in children])
        else:
            return Tree(root, [children])  # just a single terminal
    
    def logprob(self, tree):
        """ Compute the logarithmic probability of a tree in this grammar. """

        if len(tree) == 1:
            child = tree[0]
            return np.log(self[tree.head][child])
        else:
            children = tuple(branch.head for branch in tree)
            logprob = np.log(self[tree.head][children])
            logprob += sum(self.logprob(branch) for branch in tree)
            return logprob
    
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

        return expectations

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

        len_sentence, _, num_nonterminals = inside.shape
        outside = np.zeros(2*[len_sentence] + [len(self)])

        if initial is not None:  # we have not specified any starting nonterminal
            outside[0, len_sentence - 1, :] = initial
        else:
            outside[0, len_sentence - 1, :] = 1.0 / num_nonterminals

        for width in reversed(range(1, len_sentence)):
            for start in range(0, len_sentence - width + 1):
                stop = start + width
                # case 1 -- [known letters, [N: unknown letters]]: some parent
                # nonterminal generated some letters to the left of the current
                # span as well as the nonterminal heading the current span.
                for left in range(0, start):
                    parentprobs = outside[left, stop - 1, :]
                    branchprobs = self.transitions.transpose([1, 2, 0]) @ parentprobs
                    superprob = inside[left, start - 1] @ branchprobs
                    outside[start, stop - 1, :] += superprob
                # case 2 -- [[N: unknown letters], known letters]: some parent
                # nonterminal generated some letters to the right of the current
                # span as well as the nonterminal heading the current span.
                for right in range(stop + 1, len_sentence + 1):
                    parentprobs = outside[start, right - 1, :]
                    branchprobs = self.transitions.transpose([1, 2, 0]) @ parentprobs
                    superprob = inside[stop, right - 1] @ branchprobs.T
                    outside[start, stop - 1, :] += superprob
        
        return outside

    def compute_transition_probabilities(self, inside, outside):
        """ Compute conditional prob. of expansions given sentence.
        
        This returns a matrix of shape `(L, L, L)`, where `L` is the
        number of nonterminal symbols in the grammar. The entry at
        (k, i, j) is the _conditional_ probability that the nonterminal
        k will expand as (i, j) given that it occurs and given the
        characters in the sentence (and, of course, given the grammar).
        """

        len_sentence, _, num_nonterminals = inside.shape
        joints = np.zeros(3 * [num_nonterminals])
        for start in range(len_sentence):
            for stop in range(start + 2, len_sentence + 1):
                pk = outside[start, stop - 1, :][:, None, None]
                for split in range(start + 1, stop):
                    qi = inside[start, split - 1, :][None, :, None]
                    qj = inside[split, stop - 1, :][None, None, :]
                    joints += (pk * qi * qj * self.transitions)

        sums = np.sum(joints, axis=(1, 2))
        posidx, = np.where(sums > 0)
        joints[posidx, :, :] /= sums[posidx, None, None]

        return joints
    
    def compute_emission_probabilities(self, inside):

        pass


codex = {

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


if __name__ == "__main__":

    # assert grammar_is_normalized(codex)  # TODO: write normalizer
    grammar = Grammar(codex)

    # plt.plot(expected_nonterminals(grammar), "o-")
    # plt.show()

    tree = grammar.sample(root=0)
    sentence = tree.flatten()
    print("Sentence: %r\n" % sentence)
    print("Actual tree:")
    tree.pprint()
    print("(probability %.5f).\n" % np.exp(grammar.logprob(tree)))

    inside = grammar.compute_inside_probabilities(sentence)
    assert inside[0, len(sentence) - 1, 0] > 0  # root is always 0
    initial = np.array([1.] + (len(grammar) - 1)*[0.])
    outside = grammar.compute_outside_probabilities(inside, initial=initial)
    transprobs = grammar.compute_transition_probabilities(inside, outside)

    print("Actually occurred transitions:")
    transitions = defaultdict(int)
    subtrees = [tree]
    while subtrees:
        parent = subtrees.pop(0)
        if len(parent) > 1:
            left, right = parent
            transitions[parent.head, left.head, right.head] += 1
            subtrees.extend([left, right])
    for trans in transitions:
        print(trans, transitions[trans])
    print()

    print("A posteriori probable transitions:")
    for triple in zip(*np.where(transprobs)):
        print(triple, transprobs[triple])
    print()

    # print("Most probable nodes given characters below:")
    # print(np.argmax(inside, axis=2))
    # print()
    # print("Most probable nodes given all characters _not_ below:")
    # print(np.argmax(outside, axis=2))
    # print()
    # print("Actual node heads:")
    # print(tree.nodematrix())
    # print()
