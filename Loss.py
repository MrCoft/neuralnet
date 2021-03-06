import numpy as np

def safe_log(vec):
    return np.log2(vec + 1e-20)

def categorical_crossentropy(correct, predict):
    return -np.mean(correct * safe_log(predict))

def binary_crossentropy(correct, predict):
    return -np.mean(correct * safe_log(predict) + (1. - correct) * safe_log(1. - predict))

def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    dist = previous_row[-1]
    acc = 1 - dist / np.maximum(np.sum((source + target) > 0), 1)
    return acc

metrics = {
    "categorical_crossentropy": (categorical_crossentropy, 1000),
    "binary_crossentropy": (binary_crossentropy, 1000),
    "levenshtein": (levenshtein, 1000),
}
