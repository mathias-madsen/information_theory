import numpy as np


class PredictiveModel(object):

    def __init__(self, alphabet="ABC"):

        self.alphabet = sorted(alphabet)

    def conditionals(self, context):
        """ Compute the conditional distribution given the context. """

        raise NotImplementedError

    def conditional(self, context, letter):
        """ Compute P(letter | context). """

        distribution = self.conditionals(context)

        return distribution[letter]

    def cumulatives(self, context):
        """ Compute the cumulative distribution vector given the context. """

        distribution = self.conditionals(context)
        probabilities = [distribution[a] for a in self.alphabet]

        return np.cumsum([0] + probabilities)

    def cumulative(self, context, letter):
        """ Compute P(letter | context) for all letters more to the 'left'. """

        cumulatives = self.cumulatives(context)
        letter_index = self.alphabet.index(letter)

        return cumulatives[letter_index]

    def sample(self, length):
        """ Sample a sequence of the given lenghth from this random process. """

        text = ""

        for t in range(length):
            distribution = self.conditionals(text)
            letters = sorted(distribution.keys())
            probabilities = [distribution[letter] for letter in letters]
            text += np.random.choice(letters, p=probabilities)

        return text


class PolyaUrnModel(PredictiveModel):

    def conditionals(self, context):
        """ Compute the conditional distribution given the context. """

        frequencies = np.array([context.count(a) for a in self.alphabet])
        smooth_frequences = 1 + frequencies
        probabilities = smooth_frequences / np.sum(smooth_frequences)

        return dict(zip(self.alphabet, probabilities))


class MarkovModel(PredictiveModel):

    def __init__(self, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ "):

        self.alphabet = sorted(alphabet)

        alpha = np.ones(len(self.alphabet))
        self.initials = np.random.dirichlet(alpha)
        self.transitions = {letter: np.random.dirichlet(alpha)
                            for letter in self.alphabet}

    def conditionals(self, context):
        """ Compute the conditional distribution given the context. """

        if context:
            probabilities = self.transitions[context[-1]]
        else:
            probabilities = self.initials

        return dict(zip(self.alphabet, probabilities))
