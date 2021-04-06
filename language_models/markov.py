"""
Functions for approximating text structure with a Markov model.

Mostly here to provide a baseline for the other models. Training it
takes about a second or so.
"""


import numpy as np
from scipy.optimize import minimize
from collections import defaultdict


def compute_pair_frequencies(text, alphabet_size=256):
    """ Count relative freq of consecutive pairs in list of ints.
    
    Returned as an array of `(alphabet_size, alphabet_size)` where the
    first axis is the previous character, and the second axis is the
    next (predicted) character.
    """

    curr = text[1:]
    prev = text[:-1]
    pairs = np.stack([prev, curr])

    print("Counting pairs . . .")
    (char1, char2), counts = np.unique(pairs, return_counts=True, axis=1)
    print("Done.\n")

    joints = np.zeros((alphabet_size, alphabet_size))
    joints[char1, char2] = counts / np.sum(counts)
    assert np.isclose(np.sum(joints), 1.0)

    return joints


def compute_conditional_distributions(joints):
    """ From a matrix of joint pair distributions, compute conditionals.
    
    The conditional probabilities are returned in a square matrix. The
    entry at (i, j) contains the probability of character (integer) j
    given character (integer) i.
    
    When the observed probability of a condition i is zero, the
    conditional distribution is set equal to the marginal distribution.
    """

    marginals = np.sum(joints, axis=1)
    posidx = marginals > 0
    
    conditionals = joints.copy()
    conditionals[posidx, :] /= marginals[posidx, None]
    conditionals[~posidx, :] = marginals
    assert np.isclose(np.sum(conditionals), len(conditionals))
    assert np.allclose(np.sum(conditionals, axis=1), 1.0)

    return conditionals


def optimize_mixture_model_weights(text, digram_freqs):

    alphabet_size, _ = digram_freqs.shape
    monogram_freqs = np.sum(digram_freqs, axis=0)
    conditionals = compute_conditional_distributions(digram_freqs)

    curr = text[1:]
    prev = text[:-1]

    uniform_probs = np.ones_like(curr) / alphabet_size
    monogram_probs = monogram_freqs[curr,]
    digram_probs = conditionals[prev, curr]

    def unpack(params):
        params = params - np.max(params)
        alphas = np.exp(params)
        alphas /= np.sum(alphas)
        return alphas
        
    def loss(params):
        a0, a1, a2 = unpack(params)
        probs = a0*uniform_probs + a1*monogram_probs + a2*digram_probs
        print(np.mean(-np.log2(probs)))
        return np.mean(-np.log2(probs))

    print("Optimizing mixture-model weights . . .")
    result = minimize(loss, [-12., -6., 0.])
    assert result.success
    print("Done.\n")

    return unpack(result.x)


def construct_mixture_model(digram_freqs, mixture_weights):
    """ Construct a mixture of a Markov and two fallback models. """

    assert np.ndim(digram_freqs) == 2
    assert digram_freqs.shape[0] == digram_freqs.shape[1]

    assert np.shape(mixture_weights) == (3,)
    assert np.isclose(np.sum(mixture_weights), 1.0)

    w0, w1, w2 = mixture_weights

    nchars, _ = digram_freqs.shape
    joint_uniform = np.ones((nchars, nchars)) / (nchars ** 2)

    marginals = np.sum(freqs, axis=0)
    joint_monogram = marginals[:, None] * marginals[None, :]

    return w0*joint_uniform + w1*joint_monogram + w2*digram_freqs


if __name__ == "__main__":

    alphabet_size = 256

    with open("long_text.txt") as source:
        text = source.read()
        paragraphs = text.split("\n\n")
        np.random.shuffle(paragraphs)
        text = "\n\n".join(paragraphs)
        codes = [ord(c) for c in text if ord(c) < alphabet_size]

    third = len(codes) // 3
    fences = [third, 2 * third]
    subsets = np.split(codes, fences)
    # np.random.shuffle(subsets)
    train, val, test = subsets

    freqs = compute_pair_frequencies(train, alphabet_size)
    weights = optimize_mixture_model_weights(val, freqs)
    joints = construct_mixture_model(freqs, weights)
    conditionals = compute_conditional_distributions(joints)

    print("Evaluating . . . ")
    probabilities = conditionals[test[:-1], test[1:]]
    surprisals = (-1) * np.log2(probabilities)
    loss_stats = np.mean(surprisals), np.std(surprisals)
    print("Loss: %.3f ± %.3f.\n" % loss_stats)
