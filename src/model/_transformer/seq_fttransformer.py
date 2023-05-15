from .fttransformer import PositionalEncoding, FTTransformer
import torch
from torch import nn
from . import manual_activate


class SeqFTTransformer(FTTransformer):
    def __init__(
        self,
        embedding_dim,
        dropout,
        run,
        *args,
        **kwargs,
    ):
        if run and self._check_activate():
            super(SeqFTTransformer, self).__init__(
                embedding_dim=embedding_dim,
                dropout=dropout,
                *args,
                **kwargs,
            )
            self.embedding = nn.Embedding(
                num_embeddings=191, embedding_dim=embedding_dim
            )
            self.pos_encoding = PositionalEncoding(
                d_model=embedding_dim, dropout=dropout
            )
            self.run = True
        else:
            super(FTTransformer, self).__init__()
            self.run = False

    def forward(self, x, derived_tensors):
        device = torch.device(x.get_device())
        if self.run:
            seq = derived_tensors["Lay-up Sequence"].long()
            lens = derived_tensors["Number of Layers"].long()
            max_len = seq.size(1)
            # for the definition of padding_mask, see nn.MultiheadAttention.forward
            padding_mask = (
                torch.arange(max_len, device=device).expand(len(lens), max_len) >= lens
            )
            x = self.embedding(seq.long() + 90)
            x_pos = self.pos_encoding(x)
            x_trans = self.transformer(x_pos, src_key_padding_mask=padding_mask)
            x_trans = x_trans.mean(1)
            x_trans = self.transformer_head(x_trans)
            return x_trans
        else:
            return torch.zeros(x.size(0), 1, device=device)

    def _check_activate(self):
        if self._manual_activate():
            return True
        else:
            print(f"SeqTransformer module is manually deactivated.")
            return False

    def _manual_activate(self):
        return manual_activate["Seq"]
