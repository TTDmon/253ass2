import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.chord_melody_transformer_v3 import ChordMelodyTransformerV3
from tqdm import tqdm

# Load data
data = torch.load('data/processed_with_durations.pt')
pitch_seqs = data['melody_pitches']
duration_seqs = data['melody_durations']
chord_seqs = data['chord_sequences']
chord_vocab_size = len(data['chord2idx']) + 1
pitch_vocab_size = 129
duration_vocab_size = 10

# Dataset and split
dataset = TensorDataset(chord_seqs, pitch_seqs, duration_seqs)
train_len = int(0.9 * len(dataset))
train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChordMelodyTransformerV3(
    pitch_vocab_size=pitch_vocab_size,
    duration_vocab_size=duration_vocab_size,
    chord_vocab_size=chord_vocab_size
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Training
for epoch in range(10):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for chords, pitches, durations in pbar:
        chords, pitches, durations = chords.to(device), pitches.to(device), durations.to(device)
        tgt_inp = pitches[:, :-1]
        tgt_dur = durations[:, :-1]
        pitch_target = pitches[:, 1:]
        dur_target = durations[:, 1:]

        pitch_out, dur_out = model(chords, tgt_inp, tgt_dur)
        pitch_out = pitch_out.reshape(-1, pitch_vocab_size)
        dur_out = dur_out.reshape(-1, duration_vocab_size)
        pitch_target = pitch_target.reshape(-1)
        dur_target = dur_target.reshape(-1)

        loss = loss_fn(pitch_out, pitch_target) + loss_fn(dur_out, dur_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for chords, pitches, durations in val_loader:
            chords, pitches, durations = chords.to(device), pitches.to(device), durations.to(device)
            tgt_inp = pitches[:, :-1]
            tgt_dur = durations[:, :-1]
            pitch_target = pitches[:, 1:]
            dur_target = durations[:, 1:]

            pitch_out, dur_out = model(chords, tgt_inp, tgt_dur)
            pitch_out = pitch_out.reshape(-1, pitch_vocab_size)
            dur_out = dur_out.reshape(-1, duration_vocab_size)
            pitch_target = pitch_target.reshape(-1)
            dur_target = dur_target.reshape(-1)

            loss = loss_fn(pitch_out, pitch_target) + loss_fn(dur_out, dur_target)
            val_loss += loss.item()

        print(f"Validation loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), f'models/duration_model_v3.pt')
