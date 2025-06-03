import torch
import argparse
from pathlib import Path
import miditoolkit
from models.chord_melody_transformer import ChordMelodyTransformerV2
from models.popmag import ChordMelodyTransformerV3Plus

def save_melody_only(melody_ids, path, ticks_per_beat=480, note_unit=1.0):
    midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
    inst = miditoolkit.Instrument(program=0, is_drum=False, name="Melody")
    note_duration = int(ticks_per_beat * note_unit)

    for i, pitch in enumerate(melody_ids):
        if 0 < pitch < 128:
            start = i * note_duration
            end = start + note_duration
            inst.notes.append(miditoolkit.Note(velocity=80, pitch=pitch, start=start, end=end))
    midi.instruments.append(inst)
    midi.dump(path)
    print(f"✅ Saved melody-only MIDI to {path}")

def save_combined_midi(melody_ids, chord_labels, path, ticks_per_beat=480, note_unit=1.0):
    midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
    inst = miditoolkit.Instrument(program=0, is_drum=False, name="Melody")
    note_duration = int(ticks_per_beat * note_unit)

    for i, pitch in enumerate(melody_ids):
        if 0 < pitch < 128:
            start = i * note_duration
            end = start + note_duration
            inst.notes.append(miditoolkit.Note(velocity=80, pitch=pitch, start=start, end=end))
    midi.instruments.append(inst)

    for i, chord in enumerate(chord_labels):
        time = i * note_duration
        midi.markers.append(miditoolkit.Marker(text=chord, time=time))

    midi.dump(path)
    print(f"✅ Saved combined chord+melody MIDI to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chords', type=str, nargs='+', required=True, help='Chord labels (e.g. C:maj7 D:min7)')
    parser.add_argument('--model_path', type=str, default='models/best_model_2.pt')
    parser.add_argument('--data_path', type=str, default='data/processed_transposed.pt')
    parser.add_argument('--output', type=str, default='output/melody.mid')
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--start_pitch', type=int, default=60)
    parser.add_argument('--combine', action='store_true', help='Combine chords with melody')
    parser.add_argument('--note_unit', type=float, default=1.0, help='Note length in beats (e.g. 0.25 = sixteenth note)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocab
    data = torch.load(args.data_path)
    chord2idx = data['chord2idx']
    chord_vocab_size = len(chord2idx) + 1

    # Map chords
    chord_ids = [chord2idx.get(c, 0) for c in args.chords]
    chord_tensor = torch.tensor([chord_ids], dtype=torch.long, device=device)

    # Load model
    model = ChordMelodyTransformerV2(vocab_size=129, chord_vocab_size=chord_vocab_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Generate melody
    melody = [args.start_pitch]
    with torch.no_grad():
        for _ in range(args.max_length - 1):
            mel_tensor = torch.tensor([melody], dtype=torch.long, device=device)
            memory = model.encoder(model.pos_encoder(model.chord_embedding(chord_tensor)))
            tgt_emb = model.pos_encoder(model.melody_embedding(mel_tensor))
            tgt_mask = model.generate_mask(mel_tensor.size(1), device=device)
            out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = model.output_layer(out[:, -1, :]) / args.temperature
            probs = torch.softmax(logits, dim=-1)
            next_note = torch.multinomial(probs, 1).item()
            if next_note == 128:
                break
            melody.append(next_note)
    melody = melody[1:]  # remove start token

    # Save output MIDI
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    if args.combine:
        save_combined_midi(melody, args.chords, args.output, note_unit=args.note_unit)
    else:
        save_melody_only(melody, args.output, note_unit=args.note_unit)

if __name__ == "__main__":
    main()
