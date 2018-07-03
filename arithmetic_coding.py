import numpy as np


def find_outer_binary_interval(left, width):
    """ Find the code for a binary interval containing the deciaml interval. """

    bits = ""

    while True:

        if left + width <= 1/2:
            bits += "0"
            left = 2*left
            width = 2*width

        elif left >= 1/2:
            bits += "1"
            left = 2*left - 1
            width = 2*width

        else:
            break

    return bits


def find_inner_binary_interval(left, width):
    """ Find the Shannon-Fano-Elias codeword for this decimal interval. """

    num_bits = 0

    while True:

        num_steps = 2 ** num_bits
        step_width = 1 / num_steps

        for i in range(num_steps):
            step_left = i * step_width
            if left <= step_left and step_left + step_width <= left + width:
                return np.binary_repr(i, width=num_bits)

        num_bits += 1


def zoom_to_outer_binary_interval(left, width, code):
    """ Represent an inner interval as if its binary container was [0, 1]. """

    for bit in code:
        if bit == "0":
            left = 2*left
            width = 2*width
        elif bit == "1":
            left = 2*left - 1
            width = 2*width
        else:
            raise Exception("Unexpected bit: %r" % bit)

    return left, width


def find_index_of_outer_decimal(fences, bits):
    """ Find the index of the segment containing the binary interval.  """

    for w in range(len(bits) + 1):

        substring = bits[:w]
        bitint = 0 if not substring else int(bits[:w], base=2)
        bitleft = bitint / 2**w
        bitwidth = 1 / 2**w

        for i, (left, right) in enumerate(zip(fences[:-1], fences[1:])):
            if left <= bitleft and bitleft + bitwidth <= right:
                # print("match: index %s\n" % i)
                return i, substring

    for w in range(len(bits) + 1):

        substring = bits[:w]
        bitint = 0 if not substring else int(bits[:w], base=2)
        bitleft = bitint / 2**w
        bitwidth = 1 / 2**w

    return -1, ""


def encode(plaintext, model):
    """ Encode the text using a predictive model and an arithmetic encoder. """

    left = np.float64(0)
    width = np.float64(1)

    encoding = ""

    for t, letter in enumerate(plaintext):

        # commit to any bits that have already settled:
        outer_binary = find_outer_binary_interval(left, width)
        left, width = zoom_to_outer_binary_interval(left, width, outer_binary)
        encoding += outer_binary

        # subdivide the remaining interval:
        context = plaintext[:t]
        cumulatives = model.cumulatives(context)
        fences = left + width*cumulatives

        # select the interval corresponding to the next choice:
        i = model.alphabet.index(letter)
        left = fences[i]
        width = fences[i + 1] - fences[i]

    # add bits that have become certain only at the end of the file:
    encoding += find_inner_binary_interval(left, width)

    return encoding


def decode(codetext, model):
    """ Decode the text using a predictive model and an arithmetic encoder. """

    left = np.float64(0)
    width = np.float64(1)

    decoding = ""

    while True:

        outer_binary = find_outer_binary_interval(left, width)
        left, width = zoom_to_outer_binary_interval(left, width, outer_binary)
        codetext = codetext[len(outer_binary):]

        context = decoding
        cumulatives = model.cumulatives(context)
        fences = left + width*cumulatives

        i, codeword = find_index_of_outer_decimal(fences, codetext)

        if i == -1:
            return decoding

        left = fences[i]
        width = fences[i + 1] - fences[i]

        decoding += model.alphabet[i]

        if not codetext:
            return decoding


if __name__ == "__main__":

    from predictive_model import PolyaUrnModel, MarkovModel

    model = MarkovModel()
    text = model.sample(100)
    print("Encoding . . .")
    encoded = encode(text, model)
    print("Decoding . . .")
    decoded = decode(encoded, model)
    print("Comparing . . .")
    assert text == decoded
    print("Done.\n")