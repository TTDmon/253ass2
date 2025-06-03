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

class ChordMelodyTransformerV3(nn.Module):
    def __init__(self, pitch_vocab_size=129, duration_vocab_size=10, chord_vocab_size=128, d_model=256,
                 nhead=8, num_encoder_layers=4, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.chord_embedding = nn.Embedding(chord_vocab_size, d_model)
        self.pitch_embedding = nn.Embedding(pitch_vocab_size, d_model)
        self.duration_embedding = nn.Embedding(duration_vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 双输出分支
        self.pitch_out = nn.Linear(d_model, pitch_vocab_size)
        self.duration_out = nn.Linear(d_model, duration_vocab_size)

    def generate_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size), diagonal=1).to(device)
        return mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))

    def forward(self, chord_seq, pitch_input, duration_input):
        # 输入维度: [B, L]
        device = chord_seq.device

        chord_emb = self.pos_encoder(self.chord_embedding(chord_seq))
        pitch_emb = self.pitch_embedding(pitch_input)
        duration_emb = self.duration_embedding(duration_input)
        tgt_emb = self.pos_encoder(pitch_emb + duration_emb)

        memory = self.encoder(chord_emb)
        tgt_mask = self.generate_mask(pitch_input.size(1), device=device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

        pitch_logits = self.pitch_out(output)       # [B, L, pitch_vocab_size]
        duration_logits = self.duration_out(output) # [B, L, duration_vocab_size]

        return pitch_logits, duration_logits
