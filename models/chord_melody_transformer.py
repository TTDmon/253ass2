import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ChordMelodyTransformerV2(nn.Module):
    def __init__(self, vocab_size=129, chord_vocab_size=128, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.chord_embedding = nn.Embedding(chord_vocab_size, d_model)
        self.melody_embedding = nn.Embedding(vocab_size, d_model)  # +1 for EOS
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_layer = nn.Linear(d_model, vocab_size)

    def generate_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size), diagonal=1).to(device)
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask

    def forward(self, chord_seq, melody_input):
        # chord_seq: [B, L], melody_input: [B, L]
        device = chord_seq.device
        chord_emb = self.pos_encoder(self.chord_embedding(chord_seq))
        melody_emb = self.pos_encoder(self.melody_embedding(melody_input))

        memory = self.encoder(chord_emb)
        tgt_mask = self.generate_mask(melody_input.size(1), device)
        output = self.decoder(melody_emb, memory, tgt_mask=tgt_mask)

        return self.output_layer(output)  # [B, L, vocab_size]

    def generate(self, chord_seq, max_length=32, temperature=1.0, start_token=60, eos_token=128):
        self.eval()
        device = chord_seq.device
        memory = self.encoder(self.pos_encoder(self.chord_embedding(chord_seq)))

        melody = [start_token]
        for _ in range(max_length - 1):
            inp = torch.tensor([melody], dtype=torch.long, device=device)
            emb = self.pos_encoder(self.melody_embedding(inp))
            mask = self.generate_mask(inp.size(1), device)
            out = self.decoder(emb, memory, tgt_mask=mask)
            logits = self.output_layer(out[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)
            next_note = torch.multinomial(probs, 1).item()
            if next_note == eos_token:
                break
            melody.append(next_note)

        return melody[1:]  # drop start token
