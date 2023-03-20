"""
This script is adapted from FastFormer: https://github.com/wuch15/Fastformer
The github repository is not under any license (accessed 3-20-2023).
See https://arxiv.org/abs/2108.09084 for the paper.
"""


from torch import nn
import torch
from typing import *


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(hidden_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))
        return x


class FastSelfAttention(nn.Module):
    def __init__(self, embed_dim, attn_heads):
        super(FastSelfAttention, self).__init__()
        if embed_dim % attn_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (embed_dim, attn_heads)
            )
        self.attention_head_size = int(embed_dim / attn_heads)
        self.num_attention_heads = attn_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.embed_dim = embed_dim

        self.query = nn.Linear(self.embed_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.embed_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = (
            self.query_att(mixed_query_layer).transpose(1, 2)
            / self.attention_head_size**0.5
        )
        # add attention mask
        if attention_mask is not None:
            query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = (
            torch.matmul(query_weight, query_layer)
            .transpose(1, 2)
            .view(-1, 1, self.num_attention_heads * self.attention_head_size)
        )
        pooled_query_repeat = pooled_query.repeat(1, seq_len, 1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer = mixed_key_layer * pooled_query_repeat

        query_key_score = (
            self.key_att(mixed_query_key_layer) / self.attention_head_size**0.5
        ).transpose(1, 2)

        # add attention mask
        if attention_mask is not None:
            query_key_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        # query = value
        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2]
            + (self.num_attention_heads * self.attention_head_size,)
        )
        weighted_value = self.transform(weighted_value) + mixed_query_layer

        return weighted_value


class FastAttention(nn.Module):
    def __init__(self, attn_heads, embed_dim, dropout):
        super(FastAttention, self).__init__()
        self.self = FastSelfAttention(embed_dim=embed_dim, attn_heads=attn_heads)
        self.output = nn.Sequential(
            OrderedDict(
                [
                    ("dense_0", nn.Linear(embed_dim, embed_dim)),
                    ("dropout", nn.Dropout(dropout)),
                ]
            )
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, input_tensor, attention_mask=None):
        x_attn = self.self(input_tensor, attention_mask)
        attention_output = self.norm(input_tensor + self.output(x_attn))
        return attention_output


class FastformerLayer(nn.Module):
    def __init__(self, attn_heads, embed_dim, dropout, ff_dim):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(
            attn_heads=attn_heads,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.f = nn.Sequential(
            OrderedDict(
                [
                    ("dense_0", nn.Linear(embed_dim, ff_dim)),
                    ("act_func", nn.ReLU()),
                    ("dense_1", nn.Linear(ff_dim, embed_dim)),
                    ("dropout", nn.Dropout(dropout)),
                ]
            )
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, hidden_states, attention_mask=None):
        x_attn = self.attention(hidden_states, attention_mask)
        layer_output = self.norm(x_attn + self.f(x_attn))
        return layer_output


class FastformerEncoder(nn.Module):
    def __init__(
        self,
        attn_heads,
        attn_layers,
        embed_dim,
        dropout,
        ff_dim,
        max_position_embeddings,
        pooler_count=1,
    ):
        super(FastformerEncoder, self).__init__()
        self.encoders = nn.ModuleList(
            [
                FastformerLayer(
                    attn_heads=attn_heads,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    ff_dim=ff_dim,
                )
                for _ in range(attn_layers)
            ]
        )
        self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)
        self.LayerNorm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        for _ in range(pooler_count):
            self.poolers.append(AttentionPooling(hidden_size=embed_dim))

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_embs, src_key_padding_mask=None, pooler_index=0):
        # input_embs: batch_size, seq_len, emb_dim
        # attention_mask: batch_size, seq_len, emb_dim

        extended_attention_mask = None
        if src_key_padding_mask is not None:
            extended_attention_mask = src_key_padding_mask.unsqueeze(1)
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            batch_size, seq_length, emb_dim = input_embs.shape
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_embs.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.position_embeddings(position_ids)

            embeddings = input_embs + position_embeddings
        else:
            embeddings = input_embs

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # print(embeddings.size())
        all_hidden_states = [embeddings]

        for i, layer_module in enumerate(self.encoders):
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], src_key_padding_mask)

        return output
