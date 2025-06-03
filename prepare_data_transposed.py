import os
import torch
import miditoolkit
from tqdm import tqdm
from pathlib import Path

# --- Transpose chord label ---
def transpose_chord_label(chord, shift):
    import music21
    try:
        root, kind = chord.split(':')
        symbol = music21.harmony.ChordSymbol(root)
        transposed = symbol.transpose(shift)
        return f"{transposed.root().name}:{kind}"
    except:
        return chord

# --- Extract melody and chord with transposition ---
def process_song(folder, sequence_length=32, transpose_range=range(-3, 4)):
    song_id = os.path.basename(folder)
    midi_path = os.path.join(folder, f"{song_id}.mid")
    chord_path = os.path.join(folder, "chord_midi.txt")

    if not os.path.exists(midi_path) or not os.path.exists(chord_path):
        return [], [], []

    midi = miditoolkit.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    melody_track = next((inst for inst in midi.instruments if not inst.is_drum), None)
    if melody_track is None or len(melody_track.notes) == 0:
        return [], [], []

    melody_track.notes.sort(key=lambda x: x.start)
    max_tick = midi.max_tick

    # Build base melody by beat
    melody_seq = []
    for i in range(0, max_tick, ticks_per_beat):
        notes = [n.pitch for n in melody_track.notes if i <= n.start < i + ticks_per_beat]
        melody_seq.append(notes[0] if notes else 0)

    # Load chord sequence
    with open(chord_path) as f:
        chord_seq = [line.strip() for line in f.readlines()]

    min_len = min(len(melody_seq), len(chord_seq))
    melody_seq = melody_seq[:min_len]
    chord_seq = chord_seq[:min_len]

    all_melody, all_chord, chord_vocab = [], [], set()
    for shift in transpose_range:
        melody_shifted = [(p + shift if p > 0 else 0) for p in melody_seq]
        chord_shifted = [transpose_chord_label(c, shift) for c in chord_seq]
        chord_vocab.update(chord_shifted)

        for i in range(0, min_len - sequence_length + 1, sequence_length):
            all_melody.append(torch.tensor(melody_shifted[i:i+sequence_length]))
            all_chord.append(chord_shifted[i:i+sequence_length])

    return all_melody, all_chord, chord_vocab

# --- Main loop ---
DATASET_DIR = "data/POP909"
OUTPUT_PATH = "data/processed_transposed.pt"
SEQUENCE_LENGTH = 32

melody_samples, chord_samples = [], []
chord_vocab = set()

for song_id in tqdm(range(1, 910)):
    folder = os.path.join(DATASET_DIR, f"{song_id:03d}")
    melody_seq, chord_seq, vocab = process_song(folder, SEQUENCE_LENGTH)
    melody_samples.extend(melody_seq)
    chord_samples.extend(chord_seq)
    chord_vocab.update(vocab)

# Build vocab
chord_list = sorted(chord_vocab)
chord2idx = {ch: i + 1 for i, ch in enumerate(chord_list)}  # 0 for unknown
idx2chord = {i: ch for ch, i in chord2idx.items()}

# Encode chords
encoded_chords = [
    torch.tensor([chord2idx.get(c, 0) for c in seq], dtype=torch.long)
    for seq in chord_samples
]

# Save
Path("data").mkdir(parents=True, exist_ok=True)
torch.save({
    'chord_sequences': torch.stack(encoded_chords),
    'melody_sequences': torch.stack(melody_samples),
    'chord2idx': chord2idx,
    'idx2chord': idx2chord,
}, OUTPUT_PATH)

print(f"\n✅ Saved {len(encoded_chords)} samples to {OUTPUT_PATH}")
