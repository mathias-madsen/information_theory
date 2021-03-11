import numpy as np
from collections import OrderedDict, defaultdict
import re


text = open("Alice's Adventures in Wonderland.txt").read().upper()
alphabet = sorted(set(text))
words = re.split("\s+", text)

words3 = defaultdict(lambda: defaultdict(int))
words2 = defaultdict(lambda: defaultdict(int))
words1 = defaultdict(lambda: defaultdict(int))

chars3 = defaultdict(lambda: defaultdict(int))
chars2 = defaultdict(lambda: defaultdict(int))
chars1 = defaultdict(lambda: defaultdict(int))


for w1, w2, w3 in zip(words[:-2], words[1:-1], words[2:]):
    for t, char in enumerate(w3):
        words3[w1, w2, w3[:t]][char] += 1
        words2[w2, w3[:t]][char] += 1
        words1[w3[:t]][char] += 1

for c1, c2, c3 in zip(text[:-2], text[1:-1], text[2:]):
    chars3[c1 + c2][c3] += 1
    chars2[c2][c3] += 1
    chars1[""][c3] += 1


def word_conditional(context):

    words = tuple(re.split("\s+", context))
    
    if words[-3:] in words3:
        return words3[words[-3:]]
    elif words[-2:] in words2:
        return words2[words[-2:]]
    elif words[-1] in words1:
        return words1[words[-1]]
    else:
        return char_conditional(context)  # default to n-gram distribution


def char_conditional(context):

    if context[-2:] in chars3:
        return chars3[context[-2:]]
    elif context[-1:] in chars2:
        return chars3[context[-1:]]
    else:
        return chars1[""]


def conditionals(context):

    c1 = word_conditional(context)
    c2 = char_conditional(context)
    c3 = chars1[""]

    d = 1e-5
    w = [(1 - d), (1 - d)*d, (1 - d)*d**2, d**3]
    counts = [w[0]*c1[a] + w[1]*c2[a] + w[2]*c3[a] + w[3] for a in alphabet]
    probs = [count/sum(counts) for count in counts]
    
    return OrderedDict({a: prob for a, prob in zip(alphabet, probs)})


if __name__ == "__main__":

    snippet = "TEXT PROCEEDS THUSLY"

    for t, char in enumerate(snippet):
        context = snippet[:t]
        dist = conditionals(context)
        items = sorted(dist.items(), key=lambda ap: -ap[1])
        cells = "; ".join("%r: %.3f" % ap for ap in items[:4])
        print("%r" % context, "--", cells)
