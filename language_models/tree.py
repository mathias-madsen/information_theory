"""
Data structure for holding nested lists.
"""

from collections import defaultdict
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
    
    def __eq__(self, other):
        """ Check if the trees have the same node, leaves, and shape. """
        if type(other) != Tree:
            return False
        if self.head != other.head:
            return False
        if len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self, other))

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
    
    def iter_transitions(self):
        """ Yield every branching triple k --> (i, j) in the tree. """

        if len(self) == 2:
            yield self.head, self[0].head, self[1].head
            for triple in self[0].iter_transitions():
                yield triple
            for triple in self[1].iter_transitions():
                yield triple
    
    def transition_counts(self):
        """ Compile a dict of the frequency of each branching triple. """

        counts = defaultdict(int)
        for triple in self.iter_transitions():
            counts[triple] += 1
        
        return dict(counts)
    
    def iter_emissions(self):
        """ Yield every (nonterminal, terminal) emission pair. """

        if len(self) == 1:
            yield self.head, self[0]
        else:
            for pair in self[0].iter_emissions():
                yield pair
            for pair in self[1].iter_emissions():
                yield pair
    
    def emission_counts(self):
        """ Compile a dict of the frequency of each emission pair. """

        counts = defaultdict(int)
        for pair in self.iter_emissions():
            counts[pair] += 1
        
        return dict(counts)

    def spandict(self):
        """ Convert the tree into a dict of head spans. """
        
        size = self.size
        spans = {(0, size): self.head}
        
        if len(self) == 1:
            return spans
        
        left, right = self
        cut = left.size
        spans.update(left.spandict())
        spans.update({(cut + i, cut + j): head
                      for (i, j), head in right.spandict().items()})
        
        return spans
    
    def get_occupancy_matrix(self, dtype=float):
        """ Return a binary occupancy matrix, disregarding node types. """

        occupancy = np.zeros((self.size, self.size), dtype=dtype)
        
        for i, j in self.spandict().keys():
            occupancy[i, j - 1] = 1
        
        return occupancy
    
    def nodematrix(self, empty_slot_value=-1):
        """ Convert the tree into a matrix of node names. """
        
        heads = np.zeros((self.size, self.size), dtype=int)
        heads[:, :] = empty_slot_value  # for unoccupied locations

        for (i, j), head in self.spandict().items():
            heads[i, j - 1] = head

        return heads
