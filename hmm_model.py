"""
This module contains code for training and using Hidden Markov model.
"""

import numpy as np
import re


def sample(trans, emits, length=1000, context=None):
    """ Sample signal path given transition and emission params. """

    nhidden, nsignals = emits.shape

    if context:
        # assume we start the sample immediately after the context:
        forward = compute_forward(context, trans, emits)
        backward = compute_backward(context, trans, emits)
        distributions = forward * backward
        distributions /= np.sum(distributions, axis=1, keepdims=True)
        initial = distributions[-1, :]
    else:
        # assume we start the sample after a long unobserved sequence:
        initial = np.ones(nhidden) / nhidden
        for _ in range(100):
            initial = initial @ trans
    
    idx = np.random.choice(nhidden, p=initial)
    z = [idx]
    while len(z) < length:
        olddist = np.zeros(nhidden)
        olddist[idx] = 1.0
        newdist = olddist @ trans
        idx = np.random.choice(nhidden, p=newdist)
        z.append(idx)
    z = np.array(z)

    x = []
    for idx in z:
        hdist = np.zeros(nhidden)
        hdist[idx] = 1.0
        edist = hdist @ emits
        jdx = np.random.choice(nsignals, p=edist)
        x.append(jdx)
    x = np.array(x)

    return x  # we only return the observations, not the hidden states


def compute_forward(x, trans, emits):
    """ Compute the priors given all (strictly) past evidence. """

    length = len(x)
    nhidden, _ = emits.shape
    forward = np.ones(shape=(length, nhidden)) / nhidden

    # we assume that we start in a stable state given `trans`:
    prev = np.ones(nhidden) / nhidden
    for _ in range(100):
        prev = prev @ trans

    # then move forward, taking each observation into account:
    for t in range(length):
        forward[t, :] = prev @ trans
        assert np.isclose(forward[t, :].sum(), 1.0)
        # take the evidence into accout:
        prev = forward[t, :] * emits[:, x[t]]
        if np.allclose(prev, 0.0):
            prev = 1.0
        prev /= prev.sum()
    
    return forward


def compute_backward(x, trans, emits):
    """ Compute relative likelihoods of current and future obs. """

    length = len(x)
    nhidden, _ = emits.shape
    backward = np.ones(shape=(length, nhidden))

    # we start with every state having nothing speaking against it:
    curr = np.ones(nhidden)

    # we then move backwards, computing the relative
    for t in reversed(range(length)):
        curr = curr @ trans.T
        curr *= emits[:, x[t]]
        if np.allclose(curr, 0.0):
            curr[:] = 1.0
        curr /= curr.sum()
        backward[t, :] = curr

    return backward


def optimize_parameters(x, distributions):
    """ Given hidden-state distributions, choose best parameters. """

    length, nhidden = distributions.shape

    print("Optimizing emission probs . . .")
    emits = np.zeros((nhidden, nsignals))
    for t in range(length):
        emits[:, x[t]] += distributions[t, :]
    emits += 0.1 / nsignals  # virtual observations
    emits /= emits.sum(axis=1, keepdims=True)

    print("Optimizing transition probs . . .")
    oldps = distributions[:-1, :, None]
    newps = distributions[1:, None, :]
    trans = np.sum(oldps * newps, axis=0)
    trans += 0.1 / nhidden  # virtual observations
    trans /= trans.sum(axis=1, keepdims=True)

    print("Done optimizing parameters.\n")
    return emits, trans


def compute_perplexity(x, distributions, emits):
    """ Compute average negative log_2 emission likelihood of x. """

    conditionals = distributions @ emits
    indices = range(len(x))
    likelihoods = conditionals[indices, x]

    return np.mean(-np.log2(likelihoods))


# nhidden = 15
# nsignals = 128
# truetrans = np.random.dirichlet(np.ones(nhidden), size=nhidden)
# trueemits = np.random.dirichlet(np.ones(nsignals), size=nhidden)
# x = sample(truetrans, trueemits)


if __name__ == "__main__":

    with open("long_text.txt") as source:
        text = source.read()
        text = re.sub("\s+", " ", text)  # replace all whitespace by space
        # text = re.sub("\n(?!\n)", " ", text)  # un-break running text
        allx = np.array([ord(char) for char in text if ord(char) < 128])

    nhidden = 150
    nsignals = 128

    # initial parameter guesses:
    trans = np.random.dirichlet(np.ones(nhidden), size=nhidden)
    emits = np.random.dirichlet(np.ones(nsignals), size=nhidden)

    # split data up into manageable chunks; we won't split the data
    # up into train and test sets, since each chunk will be visited
    # only once, and we evaluate the loss on that chunk _before_ it
    # has a chance of having an influence on the parameter values.
    chunklength = 20000
    nchunks = len(allx) // chunklength
    fences = chunklength * np.arange(1, nchunks)
    chunks = np.split(allx, fences)
    np.random.shuffle(chunks)  # shuffle order, **in place**

    for stepidx, x in enumerate(chunks):
        print("--- Step number %s / %s ---\n" % (stepidx + 1, nchunks))
        print("Computing hidden-state distributions . . .")
        forward = compute_forward(x, trans, emits)
        backward = compute_backward(x, trans, emits)
        distributions = forward * backward
        distributions /= np.sum(distributions, axis=1, keepdims=True)
        trainloss = compute_perplexity(x, forward, emits)
        print("train loss = %.2f\n" % (trainloss,))
        emits, trans = optimize_parameters(x, distributions)
        np.savez("hmmparams.npz", emits=emits, trans=trans)

        context = [ord(letter) for letter in "CHAPTER I. "]
        codes = sample(trans, emits, context=context)
        letters = (chr(c) for c in codes)
        print("".join(letters))
        print()
