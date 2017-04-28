import numpy as np
import pickle
from Dataset import Dataset, flatten
import os

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

def demo_midi(lib, length=10):
    bpm = 120
    demo_samples = length * round(bpm / 60)

    def display(train):
        x, y = train["data_test"]

        mem = x[0].reshape((1,) + x.shape[1:])
        samples = np.zeros((demo_samples, lib.classes))

        for i in range(demo_samples):
            vec = train["model"].predict(mem)[0]
            vec = np.random.random(y.shape[1:]) < vec
            samples[i] = vec
            mem[0][0:-1] = mem[0][1:]
            mem[0][-1] = vec

        demo_file = train["output_dir"] + "/sample_{}.mid".format(train["epoch"])

        from miditime.miditime import MIDITime
        midi = MIDITime(bpm, demo_file)
        samples = lib.vector_to_piano(samples)
        midi_data = [[i, note, 127, 0.5] for i, notes in enumerate(samples) for note in notes]
        midi.add_track(midi_data)
        try:
            import sys
            stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

            midi.save_midi()
        finally:
            sys.stdout = stdout

        mp3_file = os.path.splitext(demo_file)[0] + ".mp3"
        if os.path.exists(mp3_file):
            os.remove(mp3_file)
        os.system("timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 256k {}".format(demo_file, mp3_file))

        if train["ipython"]:
            import librosa
            wave, rate = librosa.load(mp3_file)
            import matplotlib.pyplot as plt
            plt.plot(wave)
            plt.show()

            from IPython.display import Audio, display
            display(Audio(wave, rate=rate))

    return display
