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

# --- Duration class definitions ---
duration_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
def quantize_duration(dur):
    return min(range(len(duration_values)), key=lambda i: abs(duration_values[i] - dur))

# --- Extract melody and chord with transposition ---
def process_song(folder, sequence_length=32, transpose_range=range(-3, 4)):
    song_id = os.path.basename(folder)
    midi_path = os.path.join(folder, f"{song_id}.mid")
    chord_path = os.path.join(folder, "chord_midi.txt")

    if not os.path.exists(midi_path) or not os.path.exists(chord_path):
        return [], [], [], []

    midi = miditoolkit.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    melody_track = next((inst for inst in midi.instruments if not inst.is_drum), None)
    if melody_track is None or len(melody_track.notes) == 0:
        return [], [], [], []

    melody_track.notes.sort(key=lambda x: x.start)
    max_tick = midi.max_tick

    # Load chord sequence
    with open(chord_path) as f:
        chord_seq = [line.strip() for line in f.readlines()]
    chord_seq_len = len(chord_seq)

    # Extract melody notes as (pitch, duration)
    note_seq = []
    for note in melody_track.notes:
        start = note.start
        end = note.end
        pitch = note.pitch
        dur = (end - start) / ticks_per_beat
        duration_class = quantize_duration(dur)
        beat_index = int(start / ticks_per_beat)
        if beat_index < chord_seq_len:
            note_seq.append((pitch, duration_class, beat_index))

    all_pitches, all_durations, all_chords = [], [], []
    chord_vocab = set()

    for shift in transpose_range:
        pitches, durations, chords = [], [], []
        for p, d, i in note_seq:
            pitches.append(p + shift)
            durations.append(d)
            chords.append(transpose_chord_label(chord_seq[i], shift))
        chord_vocab.update(chords)

        # slice into segments
        num_segments = len(pitches) // sequence_length
        for s in range(num_segments):
            start = s * sequence_length
            end = start + sequence_length
            all_pitches.append(torch.tensor(pitches[start:end]))
            all_durations.append(torch.tensor(durations[start:end]))
            all_chords.append(chords[start:end])

    return all_pitches, all_durations, all_chords, chord_vocab

# --- Main loop ---
DATASET_DIR = "data/POP909"
OUTPUT_PATH = "data/processed_with_durations.pt"
SEQUENCE_LENGTH = 32

pitch_samples, duration_samples, chord_samples = [], [], []
chord_vocab = set()

for song_id in tqdm(range(1, 910)):
    folder = os.path.join(DATASET_DIR, f"{song_id:03d}")
    p_seq, d_seq, c_seq, vocab = process_song(folder, SEQUENCE_LENGTH)
    pitch_samples.extend(p_seq)
    duration_samples.extend(d_seq)
    chord_samples.extend(c_seq)
    chord_vocab.update(vocab)

# Build chord vocab
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
    'melody_pitches': torch.stack(pitch_samples),
    'melody_durations': torch.stack(duration_samples),
    'chord_sequences': torch.stack(encoded_chords),
    'chord2idx': chord2idx,
    'idx2chord': idx2chord,
    'duration_values': duration_values,
}, OUTPUT_PATH)

print(f"\n✅ Saved {len(encoded_chords)} samples to {OUTPUT_PATH}")
