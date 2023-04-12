import torch
from torch import nn
from . import manual_activate


class LSTM(nn.Module):
    def __init__(self, n_hidden, embedding_dim, layers, run):
        super(LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.lstm_embedding_dim = embedding_dim
        self.lstm_layers = layers
        if run and self._check_activate():
            self.seq_lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=n_hidden,
                num_layers=layers,
                batch_first=True,
            )
            # The input degree would be in range [-90, 100] (where 100 is the padding value). It will be transformed to
            # [0, 190] by adding 100, so the number of categories (vocab) will be 191
            self.embedding = nn.Embedding(
                num_embeddings=191, embedding_dim=embedding_dim
            )
            self.run = True
        else:
            self.run = False

    def forward(self, x, derived_tensors):
        device = "cpu" if x.get_device() == -1 else x.get_device()
        if self.run:
            seq = derived_tensors["Lay-up Sequence"].long()
            lens = derived_tensors["Number of Layers"].long()
            h_0 = torch.zeros(
                self.lstm_layers, seq.size(0), self.n_hidden, device=device
            )
            c_0 = torch.zeros(
                self.lstm_layers, seq.size(0), self.n_hidden, device=device
            )

            seq_embed = self.embedding(seq + 90)
            seq_packed = nn.utils.rnn.pack_padded_sequence(
                seq_embed,
                torch.flatten(lens.cpu()),
                batch_first=True,
                enforce_sorted=False,
            )
            # We don't need all hidden states for all hidden LSTM cell (which is the first returned value), but only
            # the last hidden state.
            _, (h_t, _) = self.seq_lstm(seq_packed, (h_0, c_0))
            return torch.mean(h_t, dim=[0, 2]).view(-1, 1)
        else:
            return torch.zeros(x.size(0), 1, device=device)

    def _check_activate(self):
        if self._manual_activate():
            return True
        else:
            print(f"LSTM module is manually deactivated.")
            return False

    def _manual_activate(self):
        return manual_activate["LSTM"]
