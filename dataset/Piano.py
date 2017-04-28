import numpy as np
import pickle
from Dataset import Dataset, flatten

class PianoLib:
    def __init__(self, file):
        self.library = pickle.load(open(file, "rb"))
        tones = set(flatten(self.library.values(), 3))
        self.tone_zero = min(tones)
        self.classes = max(tones) - self.tone_zero + 1
    def __getitem__(self, id):
        return PianoDataset(self, id)
    def piano_to_vector(self, song):
        data_len = len(song)
        data = np.zeros((data_len, self.classes), dtype=np.bool)
        for i in range(data_len):
            for tone in song[i]:
                data[i][tone - self.tone_zero] = 1
        return data
    def vector_to_piano(self, data):
        return [np.nonzero(tone)[0] + self.tone_zero for tone in data]

class PianoDataset(Dataset):
    def __init__(self, lib, id):
        self.data = flatten(lib.library[id])
        self.lib = lib
    def __len__(self):
        return len(self.data)
    def __getitem__(self, range):
        if isinstance(range, int): range = slice(range, range + 1)
        data = self.data[range.start:range.stop]
        data = self.lib.piano_to_vector(data)
        return data