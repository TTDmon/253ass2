import os
from pathlib import Path
import miditoolkit
import torch
from tqdm import tqdm

DATASET_DIR = 'POP909-Dataset/POP909'
SEQUENCE_LENGTH = 32
OUTPUT_PATH = 'data/processed_data.pt'

chord_vocab = set()
all_chord_seqs = []
all_melody_seqs = []

for song_id in tqdm(range(1, 910), desc='Processing songs'):
    folder = os.path.join(DATASET_DIR, f"{song_id:03d}")
    midi_path = os.path.join(folder, f"{song_id:03d}.mid")
    chord_path = os.path.join(folder, "chord_midi.txt")

    if not os.path.exists(midi_path) or not os.path.exists(chord_path):
        continue

    # Load midi
    midi = miditoolkit.MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    melody_track = None

    # Try to find main melody track (not percussion)
    for track in midi.instruments:
        if not track.is_drum:
            melody_track = track
            break

    if melody_track is None or len(melody_track.notes) == 0:
        continue

    # Sort notes by start time
    melody_track.notes.sort(key=lambda x: x.start)

    # Melody: extract pitch for every beat (step = 1 beat)
    max_tick = midi.max_tick
    melody_seq = []
    for i in range(0, max_tick, ticks_per_beat):
        notes_in_beat = [note.pitch for note in melody_track.notes if i <= note.start < i + ticks_per_beat]
        melody_seq.append(notes_in_beat[0] if notes_in_beat else 0)

    # Chord: read each beat's label
    with open(chord_path, 'r') as f:
        lines = f.readlines()
    chord_seq = []
    for line in lines:
        chord = line.strip()
        chord_seq.append(chord)
        chord_vocab.add(chord)

    # Truncate to same length
    min_len = min(len(melody_seq), len(chord_seq))
    melody_seq = melody_seq[:min_len]
    chord_seq = chord_seq[:min_len]

    # Slice into SEQUENCE_LENGTH segments
    for i in range(0, min_len - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
        melody_clip = melody_seq[i:i + SEQUENCE_LENGTH]
        chord_clip = chord_seq[i:i + SEQUENCE_LENGTH]
        all_melody_seqs.append(torch.tensor(melody_clip, dtype=torch.long))
        all_chord_seqs.append(chord_clip)

# Build chord vocab
chord_list = sorted(chord_vocab)
chord2idx = {ch: i + 1 for i, ch in enumerate(chord_list)}  # 0 reserved for unknown
idx2chord = {i: ch for ch, i in chord2idx.items()}

# Encode chords
encoded_chords = []
for chord_seq in all_chord_seqs:
    chord_ids = [chord2idx.get(ch, 0) for ch in chord_seq]
    encoded_chords.append(torch.tensor(chord_ids, dtype=torch.long))

# Stack tensors
chord_tensor = torch.stack(encoded_chords)
melody_tensor = torch.stack(all_melody_seqs)

# Save data
Path("data").mkdir(parents=True, exist_ok=True)
torch.save({
    'chord_sequences': chord_tensor,
    'melody_sequences': melody_tensor,
    'chord2idx': chord2idx,
    'idx2chord': idx2chord,
}, OUTPUT_PATH)

print(f"\n✅ Saved {len(chord_tensor)} sequences to {OUTPUT_PATH}")
print(f"Chord vocab size: {len(chord2idx)}")
