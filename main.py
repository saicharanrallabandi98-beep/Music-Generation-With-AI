import os
import numpy as np

from music21 import instrument, note, stream, chord, converter

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# =========================
# READ MIDI FILES AGAIN
# =========================

dataset_path = "dataset"

notes = []

midi_files = [f for f in os.listdir(dataset_path) if f.endswith(".mid")]

for file in midi_files:

    file_path = os.path.join(dataset_path, file)

    midi = converter.parse(file_path)

    parts = instrument.partitionByInstrument(midi)

    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:

        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

# =========================
# PREPARE DATA
# =========================

pitchnames = sorted(set(notes))

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

sequence_length = 100

network_input = []

for i in range(len(notes) - sequence_length):

    sequence_in = notes[i:i + sequence_length]

    network_input.append([note_to_int[char] for char in sequence_in])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

network_input = network_input / float(len(pitchnames))

# =========================
# LOAD TRAINED MODEL
# =========================

model = load_model("music_model.h5")

print("Model Loaded Successfully!")

# =========================
# GENERATE NOTES
# =========================

start = np.random.randint(0, len(network_input)-1)

pattern = network_input[start]

prediction_output = []

print("Generating Music...")

for note_index in range(200):

    prediction_input = np.reshape(pattern, (1, len(pattern), 1))

    prediction = model.predict(prediction_input, verbose=0)

    index = np.argmax(prediction)

    result = int_to_note[index]

    prediction_output.append(result)

    pattern = np.append(pattern, index / float(len(pitchnames)))

    pattern = pattern[1:]

# =========================
# CREATE MIDI FILE
# =========================

offset = 0

output_notes = []

for pattern in prediction_output:

    if ('.' in pattern) or pattern.isdigit():

        notes_in_chord = pattern.split('.')

        notes = []

        for current_note in notes_in_chord:

            new_note = note.Note(int(current_note))

            new_note.storedInstrument = instrument.Piano()

            notes.append(new_note)

        new_chord = chord.Chord(notes)

        new_chord.offset = offset

        output_notes.append(new_chord)

    else:

        new_note = note.Note(pattern)

        new_note.offset = offset

        new_note.storedInstrument = instrument.Piano()

        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)

midi_stream.write('midi', fp='generated_music.mid')

print("\nAI Music Generated Successfully!")
print("File Saved as: generated_music.mid")