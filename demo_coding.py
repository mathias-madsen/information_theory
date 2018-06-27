"""
LIVE CODING DEMO
----------------

This script constructs a code for an uppercase Latin alphabet, using
reasonable estimates for the letter probabilities, and then allows the
uses to watch text be encoded as it is typed into a text field.

This script was used in the 2018 NASSLLI course on information theory,

    https://www.cmu.edu/nasslli2018/courses/index.html#infotheory

For more information, contact me on mathias@micropsi-industries.com.

                                            Mathias Winther Madsen
                                            Pittsburgh, 27 June 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from huffman import build_huffman_code
from shannon import build_shannon_code
from wrappable_text_field import WrappableTextField as WrapText


# disable all key bindings for matplotlib:

for key in plt.rcParams:
	if key.startswith("keymap"):
		plt.rcParams[key] = ""


#################### CONSTRUCT CODE: ####################

alice = open("Alice's Adventures in Wonderland.txt").read().upper()

counts = defaultdict(int)
for letter in alice:
    counts[letter] += 1
total = sum(counts.values())
distribution = {letter: count/total for letter, count in counts.items()}

# choose the coding scheme you would like:
##code = build_huffman_code(distribution)
code = build_shannon_code(distribution)


#################### CREATE WINDOW: ####################

height = 7
width = 11

xmargin = .01
ymargin = .01

plt.ion()

figure = plt.figure(figsize=(width, height))

input_field = WrapText(xmargin, height - ymargin, "", ha="left", spacechar="_")
output_field = WrapText(xmargin, .7*height, "", ha="left")
score_field = WrapText(width/2, 0.05*height, "", ha="center")

plt.xlim(0, 2*xmargin + width)
plt.ylim(0, 2*ymargin + height)
plt.axis("off")


#################### MAKE KEY BINDINGS: ####################

def onclick(event):

    if event.key == "backspace":
        input_field.pop_from_source_text()
    elif event.key.upper() in code:
        input_field.append_to_source_text(event.key.upper())
    else:
        pass

    source_text = input_field.get_source_text()
    encoded_text = "".join(code[letter] for letter in source_text)
    output_field.set_source_text(encoded_text)

    if len(input_field.source_text) == 0:
        score_field.set_source_text("")
    else:
        n = len(output_field.source_text)
        m = len(input_field.source_text)
        message = "Bits per character: %.3f" % (m/n)
        score_field.set_source_text(message)
    
    plt.draw()
    plt.pause(1e-2)

figure.canvas.mpl_connect('key_press_event', onclick)


#################### RUN MAIN LOOP: ####################

while True:
    plt.pause(1e-2)
    if not plt.fignum_exists(figure.number):
        break

plt.close("all")
