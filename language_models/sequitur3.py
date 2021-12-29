UNIQUE_RULE_NUMBER = 0
DIGRAM_INDEX = dict()


def insert_into_index(pointer):
    global DIGRAM_INDEX
    # key = pointer.get_pair()
    # print("Inserting %s into the index" % (key,))
    # if key in DIGRAM_INDEX:
    #     print("This will overwrite an existing entry.")
    DIGRAM_INDEX[pointer.get_pair()] = pointer


def is_at_the_center_of_triplet(symbol):
    """ True if both neighbors exist and contain the same cargo. """
    if symbol.prev is None:
        return False
    if symbol.next is None:
        return False
    same_value = (symbol.prev.get_value() == symbol.get_value()
                                          == symbol.next.get_value())
    same_symbol = symbol.prev == symbol == symbol.next
    return same_value and not same_symbol


def is_first_token_in_a_full_rule(symbol):
    """ True for the A in X-A-B-Y, where X and Y are guards. """

    return symbol.prev.is_guard() and symbol.next.next.is_guard()


def invent_new_rule(first_symbol_of_a_digram):

    rule = Rule()

    end = rule.last()
    symbol = Symbol(first_symbol_of_a_digram)
    symbol.join(end.next)
    end.join(symbol)
    
    end = rule.last()
    symbol = Symbol(first_symbol_of_a_digram.next)
    symbol.join(end.next)
    end.join(symbol)

    return rule


class Symbol:

    def __init__(self, value):
        
        self.next = None
        self.prev = None
        self.terminal = None
        self.rule = None
        
        if type(value) == str:
            self.terminal = value
        else:
            if getattr(value, "terminal", None) is not None:
                self.terminal = value.terminal
            elif getattr(value, "rule", None) is not None:
                self.rule = value.rule
                self.rule.reference_count += 1
            else:
                self.rule = value
                self.rule.reference_count += 1

        assert self.terminal is not None or self.rule is not None

    def get_value(self):
        """ Get the name of this terminal or nonterminal. """

        return self.rule if self.rule is not None else self.terminal

    def __repr__(self):
        """ Represent this symbol as a string. """

        if self.rule is None and self.terminal is not None:
            return "<Terminal %r>" % self.terminal
        elif self.rule is not None and self.terminal is None:
            return "<Nonterminal N%s>" % self.rule.number
        else:
            raise ValueError("Unexpected symbol type.")

    def join(self, other):
        """ Append `other` to `self`, removing conflicting links. """
        
        if self.next is not None:
            if not self.is_guard() and self.next is not None and not self.next.is_guard():
                if DIGRAM_INDEX.get(self.get_pair(), None) == self:
                    DIGRAM_INDEX.pop(self.get_pair())
        
        if is_at_the_center_of_triplet(other):  # we are going to break a SSS triplet
            insert_into_index(other)  # point the index to the middle S

        if is_at_the_center_of_triplet(self):  # as above, just left conjugate
            insert_into_index(self.prev)

        self.next = other
        other.prev = self

    def delete(self):
        """ Delete this symbol, causing its neighbors to join. """

        self.prev.join(self.next)

        if not self.is_guard():
            if self.next is not None and not self.next.is_guard():
                if DIGRAM_INDEX.get(self.get_pair(), None) == self:
                    DIGRAM_INDEX.pop(self.get_pair())
            if self.rule is not None:
                self.rule.reference_count -= 1

    def is_guard(self):
        """ Whether this is the fencepost of a rule right-hand side. """

        return (self.rule is not None) and (self.rule.guard == self)

    def replace_this_nonterminal_by_its_meaning(self):
        """ Replace a last remaining use of a rule by its meaning. """

        left = self.prev
        right = self.next
        first = self.rule.first()
        last = self.rule.last()

        if DIGRAM_INDEX.get(self.get_pair(), None) == self:
            DIGRAM_INDEX.pop(self.get_pair())

        left.join(first)
        last.join(right)

        # DIGRAM_INDEX[last.get_pair()] = last
        insert_into_index(last)

    def substitute(self, rule):
        """ Replace a digram with a nonterminal representing a rule. """
        
        assert type(rule) == Rule
        rule_chars = rule.first().get_pair()
        string_chars = self.get_pair()
        assert rule_chars == string_chars, (rule_chars, string_chars)
        # print("Replacing digram %s with N%s" % (self.get_pair(), rule.number))

        prev = self.prev  # from x-A-B-y, grab x
        prev.next.delete()  # then delete A
        prev.next.delete()  # then delete B
        # print("Expansion deleted, inserting %r" % rule)
        rule_name = Symbol(rule)
        rule_name.join(prev.next)
        prev.join(rule_name)

        stop_here = False
        if prev.is_guard() or prev.next.is_guard():
            pass
        elif prev.get_pair() not in DIGRAM_INDEX:
            insert_into_index(prev)
        else:
            match = DIGRAM_INDEX[prev.get_pair()]
            assert prev != match
            if match.next != prev:
                assert prev.next != match
                prev.replace_digram_by_nonterminal(match)
                stop_here = True

        if not stop_here:
            if prev.next.is_guard() or prev.next.next.is_guard():
                pass
            elif prev.next.get_pair() not in DIGRAM_INDEX:
                insert_into_index(prev.next)
            else:
                match = DIGRAM_INDEX[self.get_pair()]
                assert prev.next != match
                if match.next != prev.next:
                    assert prev.next.next != match
                    prev.next.replace_digram_by_nonterminal(match)

    def replace_digram_by_nonterminal(self, match):
        """ Suppress a new occurrence `match` of the digram starting here. """

        if is_first_token_in_a_full_rule(match):
            rule = match.prev.rule
            self.substitute(rule)
            firstchar = rule.first()
            if firstchar.rule and firstchar.rule.reference_count < 3:
                firstchar.replace_this_nonterminal_by_its_meaning()

        else:
            rule = invent_new_rule(self)
            match.substitute(rule)
            self.substitute(rule)
            insert_into_index(rule.first())
            assert rule.reference_count == 3  # 1 definition + 2 applications
            firstchar = rule.first()
            if firstchar.rule and firstchar.rule.reference_count < 3:
                firstchar.replace_this_nonterminal_by_its_meaning()
    
    def get_pair(self):
        """ Get a string that can be used as a dict key. """

        return "%s + %s" % (self, self.next)


class Rule:

    def __init__(self):
        """ Create a new rule. """

        global UNIQUE_RULE_NUMBER
        self.reference_count = 0
        self.number = UNIQUE_RULE_NUMBER
        UNIQUE_RULE_NUMBER += 1
        self.guard = Symbol(self)
        assert self.guard.is_guard()
        self.guard.join(self.guard)
        # print("Created rule number %s, refcount %s" %
        #       (self.number, self.reference_count))

    def first(self):
        """ Return the first non-guard symbol in the rule. """

        return self.guard.next
    
    def last(self):
        """ Return the last non-guard symbol in the rule. """

        return self.guard.prev
    
    def __repr__(self):

        return "<Rule N%s>" % self.number
        
    def __iter__(self):

        num_yielded = 0
        current = self.first()
        while not current.is_guard():
            yield current
            current = current.next
            num_yielded += 1
            # if num_yielded > 1000:
            #     raise Exception("Stopped iteration early")
    
    def iterflat(self):
        """ Yield every token in the meaning of the rule. """

        for token in self:
            if token.terminal is not None:
                yield token
            elif token.rule is not None:
                for subtoken in token.rule.iterflat():
                    yield subtoken
            else:
                raise ValueError("Weird token: %r" % token)

    def expand(self):
        """ Represent the meaning of the rule as a string. """

        return "".join(leaf.terminal for leaf in self.iterflat())

    def count_tokens(self):
        """ Count the number of tokens in the meaning of the rule. """

        count = 0
        current = self.first()
        while not current.is_guard():
            count += 1
            current = current.next

        return count
    
    def compile_rulebook(self):
        """ Represent this rule and all its subrules as a grammar dict. """

        waiting = [self]
        done = []
        while waiting:
            rule = waiting.pop(0)
            for token in rule:
                if token.rule is not None:
                    waiting.append(token.rule)
            done.append(rule)
        
        return {idx: list(rule) for idx, rule in enumerate(done)}
    
    def print_grammar(self):
        """ Print a human-readable representation of the rule. """

        expansions = {self: []}
        rules = [self]
        idx = 0
        while idx < len(rules):
            for token in rules[idx]:
                if token.rule is not None and token.rule not in rules:
                    rules.append(token.rule)
                    expansions[token.rule] = []
                expansions[rules[idx]].append(token)
            idx += 1

        rulepairs = []
        for idx, rule in enumerate(rules):
            rulename = "N%s" % idx
            tokens = expansions[rule]
            strings = []
            for token in tokens:
                if token.rule is not None:
                    strings.append("N%s" % rules.index(token.rule))
                else:
                    strings.append("%s" % token.terminal)
            rulepairs.append((rulename, " ".join(strings)))

        print("\n".join("%s --> %s" % pair for pair in rulepairs))
        print()
    
    def append(self, character):
        """ Append a character to the end of start symbol. """

        symbol = Symbol(character)
        
        if self.first().is_guard():
            end = self.last()
            symbol.join(end.next)
            end.join(symbol)            
        else:
            end = self.last()
            symbol.join(end.next)
            end.join(symbol)     

            startpoint = self.last().prev
            assert not startpoint.is_guard()
            assert not startpoint.next.is_guard()
            
            if startpoint.get_pair() not in DIGRAM_INDEX:
                insert_into_index(startpoint)
            else:
                match = DIGRAM_INDEX[startpoint.get_pair()]
                assert startpoint != match
                if match.next != startpoint:
                    assert startpoint.next != match
                    startpoint.replace_digram_by_nonterminal(match)


def _test_that_the_compression_is_lossless():

    import numpy as np

    letters = np.random.choice(list("ab"), size=10000, replace=True)
    text = "".join(letters)  # also convert from `numpy.str_` to `str`

    S = Rule()
    for char in text:
        S.append(char)

    assert S.expand() == text


def _test_compression_of_balanced_binary_tree():

    text = 64 * "a"

    S = Rule()
    for char in text:
        S.append(char)

    assert S.expand() == text

    S.print_grammar()

    #  this currently looks wrong


def _test_on_string_for_which_the_javascript_implementation_fails():

    # The JavaScript implementation presented on http://www.sequitur.info/
    # compresses the string "bbbaabaaabb" into the grammar:
    #
    # 0 -> b 1 2 a 1
    # 1 -> b b
    # 2 -> b a a
    #
    # This decompresses into
    # 
    # b (b b) (b a a) a (b b) = bbbbaaabb
    # 
    # which is 9-character string unequal to the original 11-character
    # string "bbbaabaaabb". The grammar above also violates the embargo
    # against rarely used rules.
    #
    # This error is due to a one-off error in the `join` method, which
    # updates the digram indiex with a bad reference when the left-hand
    # side of the two joined strings ends in trigram like "bbb".

    text = "bbbaabaaabb"

    S = Rule()
    for char in text:
        S.append(char)

    assert S.expand() == text

    S.print_grammar()


if __name__ == "__main__":

    _test_that_the_compression_is_lossless()
    _test_compression_of_balanced_binary_tree()
    _test_on_string_for_which_the_javascript_implementation_fails()

    # import numpy as np

    # for _ in range(5):

    #     letters = np.random.choice(list("ab"), size=30, replace=True)
    #     text = "".join(letters.tolist())

    #     # text = "abbaaabbbabbbbbbbaab"  # the online version fails on this one
    #     # text = "aabbbaaabb"
    #     # text = "bbbaabaaabbabbb"  # fails deterministically
    #     # text = "bbbaabaaabb"  # fails deterministically

    #     S = Rule()
    #     for char in text:
    #         S.append(char)

    #     print(text)
    #     print(S.expand())
    #     print()

    #     S.print_grammar()
