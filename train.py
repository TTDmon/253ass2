# ---------- train_v2.py ----------
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.chord_melody_transformer import ChordMelodyTransformerV2
from music_dataset import ChordMelodyDataset
from pathlib import Path
from tqdm import tqdm


def train():
    dataset_path = 'data/processed_transposed.pt'
    batch_size = 64
    num_epochs = 25
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_data = ChordMelodyDataset(dataset_path, split='train')
    val_data = ChordMelodyDataset(dataset_path, split='val')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    chord_vocab_size = train_data.chord.max().item() + 1
    model = ChordMelodyTransformerV2(vocab_size=129, chord_vocab_size=chord_vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for chords, melody in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
            chords = chords.to(device)
            inp = melody[:, :-1].to(device)
            tgt = melody[:, 1:].to(device)
            out = model(chords, inp)
            loss = criterion(out.view(-1, 129), tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for chords, melody in val_loader:
                chords = chords.to(device)
                inp = melody[:, :-1].to(device)
                tgt = melody[:, 1:].to(device)
                out = model(chords, inp)
                loss = criterion(out.view(-1, 129), tgt.reshape(-1))
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), 'models/best_model.pt')
            print("✅ Saved best model")


if __name__ == '__main__':
    train()