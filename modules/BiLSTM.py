import torch
import torch.nn as nn


class BiLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 512, num_layers: int = 1, dropout: float = 0.3,
                 bidirectional: bool = True, rnn_type: str = 'LSTM', debug: bool = False):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.num_directions
        self.rnn_type = rnn_type
        self.debug = debug

        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

    def forward(self, src_feats: torch.Tensor, src_lens: torch.Tensor, hidden: torch.Tensor = None) -> dict:
        packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens, enforce_sorted=False)

        if hidden is not None and self.rnn_type == 'LSTM':
            half = hidden.size(0) // 2
            hidden = (hidden[:half], hidden[half:])

        packed_outputs, hidden = self.rnn(packed_emb, hidden)

        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        if self.bidirectional:
            hidden = self.concatenate_hidden_states(hidden)

        if isinstance(hidden, tuple):
            hidden = torch.cat(hidden, 0)

        return {
            "predictions": rnn_outputs,
            "hidden": hidden
        }

    def concatenate_hidden_states(self, hidden: torch.Tensor) -> torch.Tensor:

        def cat_states(h: torch.Tensor) -> torch.Tensor:
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], dim=2)

        if isinstance(hidden, tuple):
            hidden = tuple(cat_states(h) for h in hidden)
        else:
            hidden = cat_states(hidden)

        return hidden