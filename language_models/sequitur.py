"""
This script implements the SEQUITUR algorithm, a very efficient
method for compressing a text into a compact context-free grammar.

The method is described and analyzed in great detail in

    Nevill-Manning and Witten: "Identifying Hierarchical Structure in
    Sequences: A linear-time algorithm" (Journal of AI Research, 1997)
    and https://arxiv.org/pdf/cs/9709102.pdf.

For an interactive version and reference implementations in other
languages than Python, see http://www.sequitur.info/.

                                    Mathias Winther Madsen
                                    Berlin, August 2021
"""

from typing import Tuple
from typing import Union
from typing import List


class LinkedList:
    """ A data container that has a successor and predecessor. """

    def __init__(self, cargo, pred=None, succ=None) -> None:
        """ Create a single element in a linked list. """

        self.cargo = cargo
        self.pred = pred
        self.succ = succ
    
    def __repr__(self):
        return "<LinkedList with cargo=%r>" % self.cargo

    @staticmethod
    def from_list(elements):
        """ Link the elements of the list and return the first item. """

        start = previous = LinkedList(elements[0])
        for elm in elements[1:]:
            currrent = LinkedList(elm, pred=previous)
            previous.succ = currrent
            previous = currrent
        return start
    
    def __iter__(self):
        """ Iterator over all descendants of this list item. """

        curr = self
        while curr is not None:
            yield curr.cargo
            curr = curr.succ

    def last(self):
        """ Return the last item in the list, with no successors. """

        return self if self.succ is None else self.succ.last()


class Sequitur:
    """ A non-recursive grammar describing a single string. """

    def __init__(self) -> None:
        """ Create an empty grammar. """

        self.rules = {0: LinkedList(0)}
        self.index = dict()
        self.uses = dict()
    
    def feed(self, character: str) -> None:
        """ Append a single character to the string. """

        x = self.rules[0].last()
        y = LinkedList(character)
        self.join(x, y)
    
    def join(self, x: LinkedList, y: LinkedList) -> None:
        """ Link up two elements, and check the constraints. """

        # do the actual joining:
        x.succ = y
        y.pred = x
        link = (x.cargo, y.cargo)

        if x.cargo is None:
            # print("Link %r starts with None" % (link,))
            pass
        elif y.cargo is None:
            # print("Link %r ends with None" % (link,))
            pass
        elif x.pred is None:
            # print("Link %r starts with header" % (link,))
            pass
        elif link not in self.index:
            # print("Link %r previously unseen." % (link,))
            self.index[link] = x
            pass
        elif self.index[link].succ == x:
            # print("Link %r overlaps previous occurrence" % (link,))
            pass
        elif self.is_full_rule(link):
            rulenum = self.index[link].pred.cargo
            # print("Link %r matches rule %s" % (link, list(self.rules[rulenum])))
            self.apply_rule(x, rulenum)
            for symbol in link:
                self.enforce_usefulness(symbol)
        else:
            # print("Link %r occurs twice" % (link,))
            oldx = self.index[link]
            rulenum = self.make_new_rule(link)
            self.apply_rule(oldx, rulenum)
            self.apply_rule(x, rulenum)
            for symbol in link:
                self.enforce_usefulness(symbol)

    def is_full_rule(self, link: Tuple[str, str]) -> bool:
        """ True iff there is a rule with exactly this expansion. """

        ref = self.index[link].pred
        if not ref.cargo in self.rules:
            return False
        for char in link:
            ref = ref.succ
            if ref.cargo is None:
                return False
            if ref.cargo != char:
                return False
        return ref.succ is None
    
    def apply_rule(self, start: LinkedList, rulenum: int) -> None:
        """ Replace a digram starting at `start` by a nonterminal. """
        
        leftof = start.pred
        a = start
        b = start.succ
        rightof = b.succ

        # print("Replacing %r--%r with %s in %s" %
        #       (a.cargo, b.cargo, rulenum, list(leftof)))

        middle = LinkedList(rulenum)

        # print("Adding reference to new use of %r" % rulenum)
        self.uses[rulenum].add(middle)

        self.join(leftof, middle)
        if rightof is not None:
            self.join(middle, rightof)

        for (x, y) in [(leftof, a), (a, b), (b, rightof)]:
            if x is None or y is None:
                continue
            if x.cargo is None or y.cargo is None:
                continue
            link = (x.cargo, y.cargo)
            if link in self.index and self.index[link] == x:
                # print("Removing link %r--%r" % link)
                self.index.pop(link)

        for removed in [a, b]:
            if removed.cargo in self.uses:
                # print("Removing use of %r" % removed.cargo)
                self.uses[removed.cargo].remove(removed)

    def make_new_rule(self, link: Tuple[str, str]) -> int:
        """ Add a rule with the given expansion, and return its name. """

        rulenum = max(self.rules.keys()) + 1
        elements = [rulenum] + list(link)
        self.rules[rulenum] = LinkedList.from_list(elements)

        # nt, c1, c2 = self.rules[rulenum]
        # print("Made new rule %s --> %s %s" % (nt, c1, c2))

        self.index[link] = self.rules[rulenum].succ
        self.uses[rulenum] = set()

        elm = self.rules[rulenum].succ
        while elm is not None:
            if elm.cargo in self.uses:
                self.uses[elm.cargo].add(elm)
                # print("Rule %s is used %s times now"
                #       % (elm.cargo, len(self.uses[elm.cargo])))
            elm = elm.succ
        
        return rulenum
    
    def enforce_usefulness(self, symbol: Union[int, str]) -> None:
        """ If the symbol is a nonterminal, enforce its double use. """

        if symbol in self.rules:
            uses = self.uses[symbol]
            # print("Num uses of %r: %s" % (symbol, len(uses)))
            if len(uses) < 2:
                # print("Nonterminal %r violates usefulness" % symbol)
                rule = self.rules.pop(symbol)
                self.uses.pop(symbol)
                rhs = rule.succ
                use, = uses

                leftof = use.pred
                rightof = use.succ

                # print("Replacing %r with %r" % (use.cargo, list(rhs)))
                self.join(leftof, rhs)
                if rightof is not None:
                    self.join(rhs.last(), rightof)

                for (x, y) in [(leftof, use), (use, rightof)]:
                    if x is None or y is None:
                        continue
                    if x.cargo is None or y.cargo is None:
                        continue
                    link = (x.cargo, y.cargo)
                    if link in self.index and self.index[link] == x:
                        # print("Removing link %r--%r" % link)
                        self.index.pop(link)
    
    def expand(self, root: int = 0) -> List[str]:
        """ Flatten the grammar back into a string. """

        flat = []
        symbols = list(self.rules[root])
        for symbol in symbols[1:]:
            if symbol not in self.rules:
                flat.append(symbol)
            else:
                flat.extend(self.expand(symbol))
        
        return flat


def _test_agreement_with_example_from_figure_1d() -> None:
    """ Verify agreement with an example from the published paper. """

    text = "aabaaab"

    grammar = Sequitur()
    for char in text:
        grammar.feed(char)
    
    _, S0, S1, S2, S3, S4 = list(grammar.rules[0])
    assert (S1, S3, S4) == ("b", "a", "b")
    assert S0 == S2  # == 1

    _, T0, T1 = list(grammar.rules[S0])
    assert T0 == T1 == "a"


def _test_agreement_with_example_from_figure_1b() -> None:
    """ Verify agreement with an example from the published paper. """

    text = "abcdbcabcdbc"

    grammar = Sequitur()
    for char in text:
        grammar.feed(char)
    
    _, S0, S1 = list(grammar.rules[0])
    assert S0 == S1

    _, T0, T1, T2, T3 = list(grammar.rules[S0])
    assert T1 == T3
    assert T0 == "a"
    assert T2 == "d"

    _, U0, U1 = list(grammar.rules[T1])
    assert U0 == "b"
    assert U1 == "c"


PORRIDGE = """pease porridge hot,
pease porridge cold,
pease porridge in the pot,
nine days old.

some like it hot,
some like it cold,
some like it in the pot,
nine days old."""


def _test_agreement_with_pease_porridge() -> None:

    grammar = Sequitur()
    for char in PORRIDGE:
        grammar.feed(char)

    assert len(grammar.rules) == 13
    assert all(type(s) == int or s == "\n" for s in grammar.rules[0])
    assert tuple("hot") in [tuple(rule)[1:] for rule in grammar.rules.values()]
    assert tuple("old") in [tuple(rule)[1:] for rule in grammar.rules.values()]
    assert tuple("in") in [tuple(rule)[1:] for rule in grammar.rules.values()]


def _test_that_the_compression_is_lossless() -> None:

    import numpy as np

    text = np.random.choice(list("abcde"), size=100, replace=True)

    grammar = Sequitur()
    for char in text:
        grammar.feed(char)

    assert grammar.expand() == list(text)


if __name__ == "__main__":

    _test_agreement_with_example_from_figure_1b()
    _test_agreement_with_example_from_figure_1d()
    _test_agreement_with_pease_porridge()
    _test_that_the_compression_is_lossless()

    text = "abcdabcdabcd"
    grammar = Sequitur()

    print("Parsing . . .")
    for char in text:
        grammar.feed(char)
    print("Done.\n")

    print("TEXT:")
    print(text)
    print()

    print("GRAMMAR:")
    for rulenum, rule in grammar.rules.items():
        elms = list(rule)
        print(rulenum, "-->", elms[1:])
    print()

    print("USES:")
    for rule, useset in grammar.uses.items():
        print("Rule %s: used %s times" % (rule, len(useset)))
    print()
