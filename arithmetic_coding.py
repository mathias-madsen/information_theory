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



def encode(plaintext, model):
    """ Encode the given string using arithmetic coding.

    Arguments:
    ----------
    text : str
        The string to be encoded.
    model : PredictiveModel object
        An object with .conditional and .cumulative methods.
    """

    encoding = ""

    left = 0.
    width = 1.

    for t, letter in enumerate(plaintext):

        # modify the current subinterval:
        context = plaintext[:t]
        inner_width = model.conditional(context, letter)
        inner_left = model.cumulative(context, letter)
        left += width*inner_left
        width *= inner_width

        # zoom left or right as long as possible:
        while True:
            if left + width <= 0.5:
                encoding += "0"
                left = 2*left
                width = 2*width
            elif left >= 0.5:
                encoding += "1"
                left = 2*(left - 0.5)
                width = 2*width
            else:
                break

    # find the inner binary interval:
    bits = int(1 + np.ceil(-np.log2(width)))
    divisions = 2 ** bits
    for i in range(divisions):
        if left <= i/divisions and (1 + i)/divisions <= left + width:
            encoding += np.binary_repr(i, width=bits)

    return encoding


def decode(codetext, model):

    decoding = ""

    binary_left = 0.
    binary_width = 1.

    for t, bit in enumerate(codetext):

        # zoom the binary interval to its lower or upper half,
        # according to the value of the next bit in the queue:
        assert bit == "0" or bit == "1", bit
        binary_width *= 0.5
        if bit == "1":
            binary_left += binary_width

        # if the binary interval is completely contained in a predictive
        # interval or subinterval, output the corresponding letters:
        while True:
            cumulatives = model.cumulatives(decoding)
            for i in range(len(cumulatives) - 1):
                letter = model.alphabet[i]
                left = cumulatives[i]
                right = cumulatives[i + 1]
                width = right - left
                if left <= binary_left and binary_left + binary_width <= right:
                    decoding += letter
                    assert binary_width <= width, (binary_width, width)
                    binary_left = (binary_left - left) / width
                    binary_width = binary_width / width
                    break  # exit the for loop
            else:
                break  # if we did not exit the for loop, exit the while loop

    return decoding





my_model = PolyaUrnModel()
my_model = MarkovModel()

original_text = my_model.sample(80)
encoded_text = encode(original_text, my_model)
decoded_text = decode(encoded_text, my_model)

print(original_text)
print(encoded_text)
print(decoded_text)

assert original_text == decoded_text