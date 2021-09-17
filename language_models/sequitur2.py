UNIQUE_RULE_NUMBER = 0
DIGRAM_INDEX = dict()


def insert_into_index(pointer):
    global DIGRAM_INDEX
    # key = pointer.get_pair()
    # print("Inserting %s into the index" % (key,))
    # if key in DIGRAM_INDEX:
    #     print("This will overwrite an existing entry.")
    DIGRAM_INDEX[pointer.get_pair()] = pointer


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

        return self.rule if self.rule is not None else self.terminal

    def __repr__(self):

        if self.rule is None and self.terminal is not None:
            return "<Terminal %r>" % self.terminal
        elif self.rule is not None and self.terminal is None:
            return "<Nonterminal N%s>" % self.rule.number
        else:
            raise ValueError("Strange symbol.")

    def join(self, other):
        """ Append `other` to `self`, removing conflicting links. """
        
        # print("Joining %r to %r" % (self, other))

        if self.next is not None:
            # print("Deleting last digram of %s--%s--%s" %
            #       (self.prev.get_value(), self.get_value(), self.next.get_value()))
            self.delete_digram()
        
        # print("Left side: %s--%s--%s" %
        #       (self.prev.get_value() if self.prev is not None else None,
        #       self.get_value(),
        #       self.next.get_value() if self.next is not None else None))

        # print("Right side: %s--%s--%s" %
        #       (other.prev.get_value() if other.prev is not None else None,
        #       other.get_value(),
        #       other.next.get_value() if other.next is not None else None))

        if (other.prev is not None and other.next is not None and
            other.prev.get_value() == other.get_value() == other.next.get_value() and
            not (other.prev == other == other.next)):
            # print("Overlapping right-hand digrams %s" % (other.get_pair(),))
            # import ipdb; ipdb.set_trace()
            # print("Overlap 1: Inserting digram %r into the index at %s" %
            #       (other.get_pair(), other))
            insert_into_index(other)
            # DIGRAM_INDEX[other.get_pair()] = other

        if (self.prev is not None and self.next is not None and
            self.prev.get_value() == self.get_value() == self.next.get_value() and
            not (self.prev == self == self.next)):
            # print("Overlapping left-hand digrams %s" % (self.get_pair(),))
            # print("Overlap 2: Inserting digram %r into the index at %s" %
            #       (self.get_pair(), self))
            # import ipdb; ipdb.set_trace()  # HERE LIVES THE PROBLEM
            # DIGRAM_INDEX[self.get_pair()] = self
            insert_into_index(self.prev)

        self.next = other
        other.prev = self

    def delete(self):

        # print("Deleting symbol %s (preceded by %s)" %
        #       (self.get_value(), self.prev.get_value()))
        
        # import ipdb; ipdb.set_trace()
        # print("Left of deleted symbol: %r" % self.prev)
        # print("Right of deleted symbol: %r" % self.next)
        self.prev.join(self.next)

        if not self.is_guard():
            # print("The deleted symbol was right-linked to %s" % self.next)
            self.delete_digram()
            if self.rule is not None:
                # print("The deleted symbol was a rule %s" % self.rule)
                self.rule.reference_count -= 1

    def delete_digram(self):
        """ Remove the digram starting here from the index, if it's there. """

        # import ipdb; ipdb.set_trace()

        if self.is_guard() or self.next is None or self.next.is_guard():
            return
        
        # import ipdb; ipdb.set_trace()
        # print("Deleting digram %s" % (self.get_pair(),))

        if DIGRAM_INDEX.get(self.get_pair(), None) == self:
            # print("Removing link %s from the digram index" % (self.get_pair(),))
            DIGRAM_INDEX.pop(self.get_pair())

    def insert_after(self, symbol):

        # print("Linking %s--%s" % (self.get_value(), symbol.get_value()))
        # print("First the right-join: %s--%s" % (symbol.get_value(), self.next.get_value()))
        symbol.join(self.next)
        # print("Then the left-join: %s--%s" % (self.get_value(), symbol.get_value()))
        self.join(symbol)

    def is_guard(self):
        """ Whether this is the fencepost of a rule right-hand side. """

        if self.rule is None:
            return False

        return self.rule.guard == self

    def process_digram(self):
        """ Replace this digram with a rule if one applies. """

        # print("Processing digram %s" % (self.get_pair(),))

        # if self.get_pair() == "<Terminal 'b'> + <Terminal 'b'>":
        #     import ipdb; ipdb.set_trace()

        if self.is_guard():
            # print("Digram %s is at a left fence, nothing to enforce." % (self.get_pair(),))
            return False
        
        if self.next.is_guard():
            # print("Digram %s is at a right fence, nothing to enforce." % (self.get_pair(),))
            return False

        if self.get_pair() not in DIGRAM_INDEX:
            # print("Inserting %s into the index" % (self.get_pair(),))
            # DIGRAM_INDEX[self.get_pair()] = self
            insert_into_index(self)
            return False

        # print("")
        # print("Pair %s has occurred before!" % (self.get_pair(),))        
        match = DIGRAM_INDEX[self.get_pair()]
        # print("Old location context: %s %s %s %s" % (self.prev, self, self.next, self.next.next))
        # print("New location context: %s %s %s %s" % (match.prev, match, match.next, match.next.next))
        # print()
        assert self != match

        if match.next != self:
            assert self.next != match
            self.process_match(match)
            return True
        else:
            pass
            # print("...but the two occurences overlap")
    
    def expand(self):
        """ Replace a last remaining use of a rule by its meaning. """

        assert self.rule is not None
        # print("Removing underused rule %r = %r" % (self.rule, self.rule.expand()))

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
        prev.insert_after(Symbol(rule))  # link up x-R-y

        # seq = (prev.prev, prev, prev.next, prev.next.next)
        # print("The local context is now %s\n" % (seq,))
        # print("Rule N%s has now occurred %s times.\n" % (rule.number, rule.reference_count))

        # print("Processing newly created digrams: %s" % (prev.get_pair(),))
        if not prev.process_digram():
            prev.next.process_digram()

    def process_match(self, match):
        """ Suppress a new occurrence `match` of the digram starting here. """

        # print("Processing match %r" % (match.get_pair(),))

        if match.prev.is_guard() and match.next.next.is_guard():
            # print("%s is full rule" % (match.get_pair(),))
            rule = match.prev.rule
            self.substitute(rule)
        else:
            rule = Rule()
            # print("Adding contents of rule %s --" % (rule,))
            rule.last().insert_after(Symbol(self))
            rule.last().insert_after(Symbol(self.next))
            # print("--- done adding contents of rule %s.\n" % (rule,))
            # print("Substituing %s for old match %s --" % (rule, match.get_pair()))
            match.substitute(rule)
            # print("-- done substituing in rule %s.\n" % (rule,))
            # print("Substituing %s for new match %s --" % (rule, self.get_pair()))
            self.substitute(rule)
            # print("-- done substituing in rule %s.\n" % (rule,))
            # DIGRAM_INDEX[rule.first().get_pair()] = rule.first()
            insert_into_index(rule.first())
        
        firstchar = rule.first()
        if firstchar.rule and firstchar.rule.reference_count < 3:
            # print("Deleting rarely used rule N%s (used %s times)" %
            #       (firstchar, firstchar.rule.reference_count))
            firstchar.expand()
    
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

        for token in self:
            if token.terminal is not None:
                yield token
            elif token.rule is not None:
                for subtoken in token.rule.iterflat():
                    yield subtoken
            else:
                raise ValueError("Weird token: %r" % token)

    def expand(self):

        return "".join(leaf.terminal for leaf in self.iterflat())

    def count_tokens(self):

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

        symbol = Symbol(character)
        # print("S: %r\n" % S.expand())
        # print("Appending symbol: %r" % symbol.terminal)
        if self.first().is_guard():
            self.last().insert_after(symbol)
        else:
            self.last().insert_after(symbol)
            self.last().prev.process_digram()
        # print()

        # print("The grammar is now:")
        # S.print_grammar()
        # print()


def _test_that_the_compression_is_lossless():

    import numpy as np

    letters = np.random.choice(list("ab"), size=10000, replace=True)
    text = "".join(letters)  # also convert from `numpy.str_` to `str`

    S = Rule()
    for char in text:
        S.append(char)

    assert S.expand() == text


def _test_compression_of_balanced_binary_tree():

    text = 8 * "a"

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

    import numpy as np

    for _ in range(5):

        letters = np.random.choice(list("ab"), size=30, replace=True)
        text = "".join(letters.tolist())

        # text = "abbaaabbbabbbbbbbaab"  # the online version fails on this one
        # text = "aabbbaaabb"
        # text = "bbbaabaaabbabbb"  # fails deterministically
        # text = "bbbaabaaabb"  # fails deterministically

        S = Rule()
        for char in text:
            S.append(char)

        print(text)
        print(S.expand())
        print()

        S.print_grammar()
