import re
import numpy as np
import tensorflow as tf
from tensorflow.nn import softmax_cross_entropy_with_logits_v2 as xent

from matplotlib import pyplot as plt
from termcolor import colored

"""
Model:
------
input[t] = sigmoid(L.x[t] + L.hidden[t - 1] + L.cell[t - 1])
forget[t] = sigmoid(L.x[t] + L.hidden[t - 1] + L.cell[t - 1])
cell[t] = forget[t]*cell[t - 1] + input[t]*tanh(L.x[t] + L.hidden[t - 1])
output[t] = sigmoid(L.x[t] + L.hidden[t - 1] + L.cell[t])
hidden[t] = output[t]*tanh(cell[t])

Notation:
---------
L.z : a linear function of z
sigmoid(z) : tf.nn.sigmoid(z + bias)
tanh : tf.nn.tanh, the inverse tangent function

Notes:
------
The internal state of the model has two elements, `cell` and `hidden`.
Both of those vectors are passed on from step to step.
"""

# def predstep(old_state, observation):
#     """ Simple implementation of the LSTM recurrence relation. """
#     old_cell, old_hidden = old_state
#     dim = old_cell.shape[-1]
#     fatstack = tf.concat([observation, old_hidden, old_cell], axis=-1)
#     thinstack = tf.concat([observation, old_hidden], axis=-1)
#     igate = tf.layers.dense(fatstack, units=dim, activation=tf.nn.sigmoid)
#     fgate = tf.layers.dense(fatstack, units=dim, activation=tf.nn.sigmoid)
#     repl_cell = tf.layers.dense(thinstack, units=dim, activation=tf.nn.tanh)
#     new_cell = fgate*old_cell + igate*repl_cell
#     newstack = tf.concat([observation, old_hidden, new_cell], axis=-1)
#     ogate = tf.layers.dense(newstack, units=dim, activation=tf.nn.sigmoid)
#     new_hidden = ogate*tf.nn.tanh(new_cell)
#     return (new_cell, new_hidden)


class SigmoidFunction:
    def __init__(self, outdim, skew=0.0):
        self.outdim = outdim
        self.skew = tf.Variable(skew, dtype=tf.float32)
        self.f1 = tf.layers.Dense(outdim, activation=tf.nn.relu)
        self.f2 = tf.layers.Dense(outdim, activation=tf.nn.relu)
        self.f3 = tf.layers.Dense(outdim, activation=tf.nn.relu)
    def __call__(self, vectors):
        stack = tf.concat(vectors, axis=-1)
        linear = self.f3(self.f2(self.f1(stack)))
        return tf.nn.sigmoid(linear + self.skew)


class MLP:
    def __init__(self, outdim, last_activation=None):
        self.f1 = tf.layers.Dense(outdim, activation=tf.nn.relu)
        self.f2 = tf.layers.Dense(outdim, activation=tf.nn.relu)
        self.f3 = tf.layers.Dense(outdim, activation=last_activation)
    def __call__(self, inputs):
        stack = tf.concat(inputs, axis=-1)
        return self.f3(self.f2(self.f1(stack)))


dim_hidden = 400

retention = SigmoidFunction(dim_hidden, +0.0)
replacement = SigmoidFunction(dim_hidden, +0.0)
outsquash = SigmoidFunction(dim_hidden, 0.0)
suggestion = MLP(dim_hidden, last_activation=tf.nn.tanh)
predict = MLP(128, last_activation=None)


def compute_next_state(old_cell, old_hidden, observation):
    # compute the suggested replacement for the cell contents:
    repl_cell = suggestion([observation, old_hidden])
    # Decide how much much of the old and suggested cell contents to mix in:
    retain = retention([observation, old_hidden, old_cell])
    replace = replacement([observation, old_hidden, old_cell])
    new_cell = retain*old_cell + replace*repl_cell
    # Decide how much the hidden state should be zero vs. tanh(cell):
    ogate = outsquash([observation, old_hidden, new_cell])
    new_hidden = ogate * tf.nn.tanh(new_cell)
    return new_cell, new_hidden


def predstep(old_state, observation):
    """ An LSTM recurrence relation with adjustable retention rate. """
    old_cell, old_hidden = old_state
    new_cell, new_hidden = compute_next_state(old_cell, old_hidden, observation)
    return new_cell, new_hidden


# trainable initial states of the chain:
initial_cell = tf.Variable(np.zeros([1, dim_hidden]), dtype=tf.float32)
initial_hidden = tf.Variable(np.zeros([1, dim_hidden]), dtype=tf.float32)

integers = tf.placeholder(shape=[None], dtype=tf.int32)
encodings = tf.one_hot(indices=integers, depth=128, dtype=tf.float32)
observations = encodings[:, None, :]

pred_initial = (initial_cell, initial_hidden)
cells, hiddens = tf.scan(predstep, elems=observations, initializer=pred_initial)

cells = cells[:, 0, :]
hiddens = hiddens[:, 0, :]
predictions = predict([cells, hiddens])

losses_in_nats = xent(labels=encodings[1:], logits=predictions[:-1])
losses_in_bits = 1 / np.log(2.0) * losses_in_nats
meanloss = tf.reduce_mean(losses_in_bits)
learning_rate = tf.placeholder(shape=[], dtype=tf.float32)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
improve = optimizer.minimize(meanloss)


def samplestep(old_state, _):
    old_cell, old_hidden, old_idx = old_state
    observation = tf.one_hot([old_idx], depth=128, dtype=tf.float32)
    new_cell, new_hidden = compute_next_state(old_cell, old_hidden, observation)
    logits = predict([new_cell, new_hidden])
    new_idx = tf.squeeze(tf.random.categorical(logits, 1))
    return new_cell, new_hidden, new_idx

initial_idx = tf.zeros([], dtype=tf.int64)
sample_initial = (initial_cell, initial_hidden, initial_idx)
_, _, sampled_idx = tf.scan(samplestep, elems=tf.range(1000), initializer=sample_initial)


if __name__ == "__main__":

    with open("long_text.txt") as source:
        text = source.read()
        # because we are training for screen display, we get rid
        # of all newline characters and other unprintable elements:
        text = re.sub("\s+", " ", text)
        alphabet = [chr(i) for i in range(128) if chr(i).isprintable()]
        text = "".join(char for char in text if char in alphabet)
        fences = [m.end() for m in re.finditer("[.!?] ", text)]
        corpus = np.array([ord(char) for char in text])

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    snippet_length = 200
    fences = [s for s in fences if s + snippet_length <= len(corpus)]

    learning_rates = 1000*[0.1] + 2000*[0.01] + 3000*[0.001] + 4000*[0.0001]
    # learning_rates = 1000*[1e-4]
    recent_losses = []
    for epochidx, lrate in enumerate(learning_rates):
        start = np.random.choice(fences)
        end = start + snippet_length
        snippet = corpus[start : end]
        feed = {integers: snippet, learning_rate: lrate}
        loss, _ = session.run([meanloss, improve], feed)
        recent_losses.append(loss)
        print("%s: %.3f\r" % (epochidx, loss), end="")
        if (epochidx + 1) % 100 == 0:
            # --- average the losses since the last printing ---
            print("Average loss: %.3f\n" % np.mean(recent_losses))
            recent_losses = []
            # --- visualize model state by printing a sample ---
            codes = session.run(sampled_idx)
            characters = "".join(chr(k) for k in codes)
            print(characters, "\n")
            # --- visualize prediction difficulty by color-printing ---
            losses = session.run(losses_in_bits, {integers: snippet})
            def paint(idx, loss):
                if loss < 2:
                    return colored(chr(idx), "green")
                elif loss < 4:
                    return colored(chr(idx), "yellow")
                else:
                    return colored(chr(idx), "red")
            colortext = "".join(paint(*pair) for pair in zip(snippet, losses))
            print(colortext, "\n")

# text = "Single-layer perceptrons are only capable of learning linearly separable patterns. For a classification task with some step activation function, a single node will have a single line dividing the data points forming the patterns. More nodes can create more dividing lines, but those lines must somehow be combined to form more complex classifications. A second layer of perceptrons, or even linear nodes, are sufficient to solve a lot of otherwise non-separable problems."
# codes = [ord(character) for character in text]
# c, h = session.run([ungated, hiddens], {integers: codes})
# figure, (top, bot) = plt.subplots(figsize=(12, 8), nrows=2, sharex=True)
# top.imshow(c.T, aspect="auto")
# bot.imshow(h.T, aspect="auto")
# plt.tight_layout()
# plt.show()