"""
Code for training hidden Markov models using the EM algorithm.

Training this model takes about the time it takes to boil a pot of pasta.
The trained models are about 15% better than a plain Markov model.

A representative sample from a trained model looks like this:

 ... Ingh ceptte thery prelve I a hice onl hary hakessemine therstl
 it thas coug thind coms, ther we whansh set rines suraticapra pewn
 sust thas allalle ting anghtt? I fom on mant inscor mur, limb that
 dents thin, a peoug. The on wition butmamp an of counther predist
 they as mok us alardliblersen thas mem on of sty hasoch hilly hictals
 hexcan ho muf anion was. Whe Wertffe domel plete, of tunouinfath. Ilkd
 busiof ther ang on. Mach leth trict ing of dicaly bust dable empl aise
 spruly of mon. subered a proull padie hen! af wher dot ingh ang bethe
 my shontrar offusher the wist NV Britord, oned dlies thoore rok his
 pusoffach to by innot be witiofferpron earder mider and wat laings,
 any besdinaked ind has sundzandeve comed was was becian ous cores And
 as had galich by coulatte. ...

which is not English, but also not not English.
"""

import numpy as np
import re


nhidden = 120
nsignals = 128
chunklength = 10000
chunks_per_update = 10


def draw_sample(trans, emits, length):
    """ Draw a sample from a hidden Markov model. """
    
    nhidden, nsignals = emits.shape
    assert trans.shape[0] == trans.shape[1] == nhidden

    dist = equilibrium(trans)
    newh = np.random.choice(nhidden, p=dist)
    hiddens = [newh]
    for _ in range(length - 1):
        oldh = hiddens[-1]
        newh = np.random.choice(nhidden, p=trans[oldh, :])
        hiddens.append(newh)

    observations = [np.random.choice(nsignals, p=emits[h]) for h in hiddens]
    
    return observations, hiddens


def equilibrium(trans):
    """ Find a stochastic vector `v` such that `v @ trans == v`. """
    assert np.allclose(np.sum(trans, axis=1), 1.0)
    values, vectors = np.linalg.eig(trans.T)
    unit_idx = np.argmin(np.abs(values - 1.0))
    stationary = np.real(vectors[:, unit_idx])
    stationary /= np.sum(stationary)
    assert np.allclose(stationary @ trans, stationary)
    return stationary


def monogram_probs(trans, emits):
    """ Compute the output distribution in the model's steady state. """
    return equilibrium(trans) @ emits


def bigram_probs(trans, emits):
    """ Compute the _pair_ distribution in the model's steady state.
    
    The probabilities come in a table of shape `[nsignals, nsignals]`.
    The entry at `(i, j)` contains the probability of observing the
    character pair `(X1 = i, X2 = j)`.
    """
    hdist = equilibrium(trans)          # P(Z1 = i), shape (K,)
    hhdist = hdist[:, None] * trans     # P(Z1 = i, Z2 = j), shape (K, K)
    hedist = hhdist @ emits             # P(Z1 = i, X2 = t), shape (K, S)
    eedist = (hedist.T @ emits).T       # P(X1 = s, X2 = t), shape (S, S)
    return eedist


def convert_text_to_array_of_integers(text, ord_cutoff=128):
    """ Remove exotic characters and certain types of whitespace. """
    text = re.sub("\s+", " ", text)  # replace all whitespace by space
    ords = [ord(char) for char in text.strip() if ord(char) < ord_cutoff]
    return np.array(ords)


def compute_kullback_divergece(p, q):
    """ Compute the KL divergence from a stochastic array `p` to `q`. """
    P = p[p > 0]
    Q = q[p > 0]
    return np.mean(P * np.log2(P / Q))


def compute_monogram_frequencies(integers, nsignals):
    """ Compile relative frequencies of all integers in `range(nsignals)`. """
    assert np.all(integers < nsignals)
    freqs = np.zeros(nsignals)
    values, counts = np.unique(integers, return_counts=True)
    freqs[values] += counts / np.sum(counts)
    return freqs


def compute_digram_frequencies(integers, nsignals):
    """ Compile digram freqencies of consecutive integers the sample. """
    freqs = np.zeros((nsignals, nsignals))
    for i, j in zip(integers[:-1], integers[1:]):
        freqs[i, j] += 1
    freqs /= np.sum(freqs)
    return freqs


def compute_conditional_entropy(pair_probabilities):
    """ Compute the average negative log-probability of a Markov model. """
    margin = np.sum(pair_probabilities, axis=1)
    baseline = margin[:, None] * margin[None, :]
    baseline += 1e-5
    joints = pair_probabilities + 1e-5*baseline
    joints /= np.sum(joints)
    conds = joints / np.sum(joints, axis=1, keepdims=True)
    return (-1) * np.sum(freqs2 * np.log2(conds))


print("Loading text . . .")
with open("long_text.txt") as source:
    raw = source.read()
    observation = convert_text_to_array_of_integers(raw, nsignals)
print("Done; the text contains %s characters." % len(observation))
print("")

# Compute the empirical frequencies, so that we
# can compare with a Markov model as a baseline:
freqs1 = compute_monogram_frequencies(observation, nsignals)
freqs2 = compute_digram_frequencies(observation, nsignals)
entropy = compute_conditional_entropy(freqs2)
meanprob = 0.5 ** entropy
print("Markov model baseline:")
print("H(X_t | X_{t - 1}) = %.5f = -log(%.5f)." % (entropy, meanprob))
print("")

# Divide the corpus up into smaller chunks:
nchunks = len(observation) // chunklength
fences = chunklength * np.arange(1, nchunks)
chunks = np.split(observation, fences)
np.random.shuffle(chunks)  # shuffle order, **in place**
bounds = range(0, len(chunks), chunks_per_update)
subsets = [chunks[b : b + chunks_per_update] for b in bounds]

# initialize parameters:
trans = np.random.dirichlet(np.ones(nhidden), size=nhidden)
emits = np.random.dirichlet(np.ones(nsignals), size=nhidden)

# compute fallback parameters used in the absense of data:
fallback_trans = np.ones((nhidden, nhidden)) / nhidden
fallback_emits = np.ones((nhidden, nsignals)) * freqs1

# create accumulators that will contain the
# stats required for the next parameter update:
transition_counter = np.zeros_like(trans)
emission_counter = np.zeros_like(emits)
recent_losses = []

for stepidx, subset in enumerate(subsets):

    for snippet in subset:

        snippet_length = len(snippet)
        likelihoods = emits[:, snippet].T

        exclusive_past = np.zeros((snippet_length, nhidden))
        # inclusive_past = np.zeros((snippet_length, nhidden))
        curr = np.ones(nhidden) / nhidden
        for t, like in enumerate(likelihoods):
            exclusive_past[t, :] = curr
            curr *= like
            curr /= curr.sum()
            # inclusive_past[t, :] = curr
            curr = curr @ trans

        # exclusive_future = np.ones((snippet_length, nhidden))
        inclusive_future = np.ones((snippet_length, nhidden))
        curr = equilibrium(trans)
        for t in reversed(range(len(snippet))):
            # exclusive_future[t, :] = curr
            like = likelihoods[t, :]
            curr *= like
            curr /= curr.sum()  # destroy true likelihood, but good for stability
            inclusive_future[t, :] = curr
            curr = curr @ trans.T

        inclusive_past = exclusive_past * likelihoods
        inclusive_past /= np.sum(inclusive_past, axis=1, keepdims=True)

        kl1 = compute_kullback_divergece(freqs1, monogram_probs(trans, emits))
        kl2 = compute_kullback_divergece(freqs2, bigram_probs(trans, emits))
        marginals = exclusive_past[:, None, :] @ likelihoods[:, :, None]
        marginals = np.squeeze(marginals, axis=(1, 2))
        loss = np.mean(-np.log2(marginals))
        recent_losses.append(loss)
        lines = [
            ("Step %s --" % stepidx).rjust(12),
            "loss: %.5f = -log_2(%.5f);" % (loss, 0.5 ** loss),
            "KL to Markov model: %.5f (monogram), %.5f (digram)" % (kl1, kl2)
        ]
        print(" ".join(lines))

        posteriors = exclusive_past * inclusive_future
        for t, xt in enumerate(snippet):
            emission_counter[:, xt] += posteriors[t, :]    

        # print("--> half done -->")
        joint = inclusive_past[:-1, :, None] * inclusive_future[1:, None, :]
        joint *= trans
        joint /= np.sum(joint, axis=(1, 2), keepdims=True)
        transition_counter += np.sum(joint, axis=0)
        # print("--> Done updating event counters.\n\n")

    new_emits = emission_counter / np.sum(emission_counter, axis=1, keepdims=True)
    new_trans = transition_counter / np.sum(transition_counter, axis=1, keepdims=True)
    emission_counter *= 0
    transition_counter *= 0

    # If we were clever about this, we would keep track of how much the
    # parameters and therefore the conditional distributions had changed
    # since last parameter update, and this would give us information
    # about how seriously we could take evidence collected under the old
    # posterior distribution; once we started converging to a local max
    # of the model, this would lead to a behavior where evidence was
    # simply accumulated across epochs, rather than erased by each update.
    # But as it is, we forget old evidence exponentially fast.
    weights = np.array([1.0, 1e-2, 1e-5])  # new, old, fallback
    weights /= weights.sum()
    trans = weights[0]*transition_counter + weights[1]*trans + weights[2]*fallback_trans
    emits = weights[0]*emission_counter + weights[1]*emits + weights[2]*fallback_emits

    np.savez("hmmparams.npz", emits=emits, trans=trans)

    print("")
    print("Mean of recent losses: %.5f.\n" % np.mean(recent_losses))
    recent_losses = []

    sample, _ = draw_sample(trans, emits, 1000)
    letters = [chr(xt) for xt in sample if xt < 128]
    print("".join(c for c in letters if c.isprintable()))
    print("")

