import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ChordMelodyTransformerV3Plus(nn.Module):
    def __init__(self, vocab_size=129, chord_vocab_size=128, d_model=256,
                 nhead=4, num_layers=3, hidden_size=256, dropout=0.1):
        super().__init__()

        # Embeddings
        self.chord_emb = nn.Embedding(chord_vocab_size, d_model)
        self.melody_emb = nn.Embedding(vocab_size, d_model)

        # Positional encodings
        self.chord_pos = PositionalEncoding(d_model)
        self.melody_pos = PositionalEncoding(d_model)

        # Transformer encoder for chords
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention layer: query from melody GRU, key/value from chords
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        # GRU decoder (2 layers)
        self.gru = nn.GRU(d_model, hidden_size, num_layers=2, batch_first=True)

        # Output projection
        self.out = nn.Linear(hidden_size, vocab_size)

        # Gating
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, chord_seq, melody_input):
        """
        chord_seq: [B, L], melody_input: [B, L]
        """
        device = chord_seq.device
        chord_embed = self.chord_pos(self.chord_emb(chord_seq))  # [B, L, D]
        chord_memory = self.encoder(chord_embed)  # [B, L, D]

        melody_embed = self.melody_pos(self.melody_emb(melody_input))  # [B, L, D]

        # Cross-attention at every step
        attn_output, _ = self.attn(melody_embed, chord_memory, chord_memory)  # [B, L, D]

        # Gated fusion: combine attention output + melody_embed
        gate_val = self.gate(torch.cat([melody_embed, attn_output], dim=-1))
        fused = gate_val * melody_embed + (1 - gate_val) * attn_output  # [B, L, D]

        output, _ = self.gru(fused)
        return self.out(output)

    def generate(self, chord_seq, max_length=32, temperature=1.0, start_token=60, eos_token=128):
        self.eval()
        device = chord_seq.device

        chord_embed = self.chord_pos(self.chord_emb(chord_seq))
        chord_memory = self.encoder(chord_embed)

        hidden = None
        melody = [start_token]
        for _ in range(max_length - 1):
            prev_note = torch.tensor([[melody[-1]]], dtype=torch.long, device=device)
            note_embed = self.melody_pos(self.melody_emb(prev_note))

            attn_out, _ = self.attn(note_embed, chord_memory, chord_memory)
            gate_val = self.gate(torch.cat([note_embed, attn_out], dim=-1))
            fused = gate_val * note_embed + (1 - gate_val) * attn_out

            out, hidden = self.gru(fused, hidden)
            logits = self.out(out[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_note = torch.multinomial(probs, 1).item()
            if next_note == eos_token:
                break
            melody.append(next_note)

        return melody[1:]
