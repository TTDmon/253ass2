import torch
import argparse
import miditoolkit
from pathlib import Path
from models.chord_melody_transformer_v3 import ChordMelodyTransformerV3

def save_combined_midi(pitches, durations, chords, out_path, ticks_per_beat=480):
    midi = miditoolkit.MidiFile(ticks_per_beat=ticks_per_beat)
    inst = miditoolkit.Instrument(program=0, is_drum=False, name="Melody")
    tick = 0
    for p, d in zip(pitches, durations):
        if p > 0 and p < 128:
            dur_tick = d * ticks_per_beat
            inst.notes.append(miditoolkit.Note(velocity=80, pitch=p, start=tick, end=tick+dur_tick))
            tick += dur_tick
    midi.instruments.append(inst)

    for i, chord in enumerate(chords):
        midi.markers.append(miditoolkit.Marker(text=chord, time=i*ticks_per_beat))

    midi.dump(out_path)
    print(f"✅ Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chords', type=str, nargs='+', required=True)
    parser.add_argument('--model_path', type=str, default='models/duration_model_v3.pt')
    parser.add_argument('--data_path', type=str, default='data/processed_with_durations.pt')
    parser.add_argument('--output', type=str, default='output/dual_output.mid')
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(args.data_path)
    chord2idx = data['chord2idx']
    chord_vocab_size = len(chord2idx) + 1

    chord_ids = [chord2idx.get(c, 0) for c in args.chords]
    chord_tensor = torch.tensor([chord_ids], dtype=torch.long, device=device)

    model = ChordMelodyTransformerV3(
        pitch_vocab_size=129, duration_vocab_size=10, chord_vocab_size=chord_vocab_size
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    pitch_seq = [60]
    dur_seq = [2]
    with torch.no_grad():
        memory = model.encoder(model.pos_encoder(model.chord_embedding(chord_tensor)))
        for _ in range(args.max_length - 1):
            pitch_inp = torch.tensor([pitch_seq], dtype=torch.long, device=device)
            dur_inp = torch.tensor([dur_seq], dtype=torch.long, device=device)

            emb = model.pos_encoder(
                model.pitch_embedding(pitch_inp) + model.duration_embedding(dur_inp)
            )
            mask = model.generate_mask(pitch_inp.size(1), device)
            out = model.decoder(emb, memory, tgt_mask=mask)

            pitch_logits = model.pitch_out(out[:, -1, :]) / args.temperature
            dur_logits = model.duration_out(out[:, -1, :]) / args.temperature

            pitch_next = torch.multinomial(torch.softmax(pitch_logits, dim=-1), 1).item()
            dur_next = torch.multinomial(torch.softmax(dur_logits, dim=-1), 1).item()

            if pitch_next == 128:
                break
            pitch_seq.append(pitch_next)
            dur_seq.append(dur_next)

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    save_combined_midi(pitch_seq[1:], dur_seq[1:], args.chords, args.output)

if __name__ == "__main__":
    main()
