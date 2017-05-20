# neuralnet
A collection of tools to do interesting stuff with Keras smoothly

# Required Python packages
* natsort
* pillow
* miditime
* librosa

### Features

# Dataset types

# Text dataset
dataset.Char.CharLib 
* Decodes using UTF-8, ignores errors.
* Turns the text to lowercase.
* Gathers a sorted list of all used characters.
* Provides useful chars_to_vector and vector_to_chars functions. They are methods of CharLib so you don't have to pass "classes" to them.
* I keep a couple of .txt datasets in one directory

# Train loop control
Displays: Functions 

# RNNs
Guided predict: Feeds the network correct samples, 
Raw predict: The probabilities

Demo: Starts with 1 sample
