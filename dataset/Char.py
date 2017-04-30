import numpy as np
import glob
import os
from Dataset import Dataset

class CharLib:
    def __init__(self, dir):
        self.library = {}
        charset = set()
        for file in glob.glob(dir + "/*"):
            text = open(file, encoding="utf-8", errors="ignore").read()
            text = text.lower()
            charset |= set(text)
            self.library[os.path.splitext(os.path.basename(file))[0]] = text
        self.classes = len(charset)
        self.alphabet = sorted(charset)
        self.labels = {c: i for i, c in enumerate(self.alphabet)}
    def __getitem__(self, id):
        return CharDataset(self, id)
    def chars_to_vector(self, text):
        data_len = len(text)
        data = np.zeros((data_len, self.classes), dtype=np.bool)
        for i, chr in enumerate(text):
            data[i][self.labels[chr]] = 1
        return data
    def vector_to_chars(self, data):
        text = ""
        for choice in np.nonzero(data)[1]:
            text += self.alphabet[choice]
        return text

class CharDataset(Dataset):
    def __init__(self, lib, id):
        self.data = lib.library[id]
        self.lib = lib
    def __len__(self):
        return len(self.data)
    def __getitem__(self, range):
        if isinstance(range, int): range = slice(range, range + 1)
        data = self.data[range.start:range.stop]
        data = self.lib.chars_to_vector(data)
        return data

def demo_text(lib, demo_samples=500):
    def display(train):
        x, y = train["data_test"]

        mem = x[0].copy()
        samples = np.zeros((demo_samples, lib.classes))

        for i in range(demo_samples):
            vec = train["model"].predict(np.expand_dims(mem, axis=0))[0]
            choice = np.random.choice(lib.classes, p=vec/np.sum(vec))
            vec = np.zeros(lib.classes)
            vec[choice] = 1
            samples[i] = vec
            mem[0:-1] = mem[1:]
            mem[-1] = vec

        text = lib.vector_to_chars(samples)
        print(text)
        with open(train["output_dir"] + "/sample_{}.txt".format(train["epoch"]), "w") as file:
            file.write(text)

    return display
