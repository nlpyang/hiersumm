"""
Implementation of "Attention is All You Need"
"""
import math

import torch.nn as nn
import torch

from abstractive.attn import MultiHeadedAttention, MultiHeadedPooling
from abstractive.neural import PositionwiseFeedForward, PositionalEncoding, sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerPoolingLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerPoolingLayer, self).__init__()

        self.pooling_attn = MultiHeadedPooling(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        context = self.pooling_attn(inputs, inputs,
                                    mask=mask)
        out = self.dropout(context)

        return self.feed_forward(out)


class TransformerInterEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, inter_layers, inter_heads, device):
        super(TransformerInterEncoder, self).__init__()
        inter_layers = [int(i) for i in inter_layers]
        self.device = device
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim / 2))
        self.dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList(
            [TransformerInterLayer(d_model, inter_heads, d_ff, dropout) if i in inter_layers else TransformerEncoderLayer(
                d_model, heads, d_ff, dropout)
             for i in range(num_layers)])
        self.transformer_types = ['inter' if i in inter_layers else 'local' for i in range(num_layers)]
        print(self.transformer_types)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_blocks, n_tokens = src.size()
        # src = src.view(batch_size * n_blocks, n_tokens)
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0

        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        inter_pos_emb = self.pos_emb.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        combined_pos_emb = torch.cat([local_pos_emb, inter_pos_emb], -1)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_layers):
            if (self.transformer_types[i] == 'local'):
                word_vec = self.transformer_layers[i](word_vec, word_vec,
                                                      1 - mask_local)  # all_sents * max_tokens * dim
            elif (self.transformer_types[i] == 'inter'):
                word_vec = self.transformer_layers[i](word_vec, 1 - mask_local, 1 - mask_block, batch_size, n_blocks)  # all_sents * max_tokens * dim

        word_vec = self.layer_norm(word_vec)
        mask_hier = mask_local[:, :, None].float()
        src_features = word_vec * mask_hier
        src_features = src_features.view(batch_size, n_blocks * n_tokens, -1)
        src_features = src_features.transpose(0, 1).contiguous()  # src_len, batch_size, hidden_dim
        mask_hier = mask_hier.view(batch_size, n_blocks * n_tokens, -1)
        mask_hier = mask_hier.transpose(0, 1).contiguous()

        unpadded = [torch.masked_select(src_features[:, i], mask_hier[:, i].byte()).view([-1, src_features.size(-1)])
                    for i in range(src_features.size(1))]
        max_l = max([p.size(0) for p in unpadded])
        mask_hier = sequence_mask(torch.tensor([p.size(0) for p in unpadded]), max_l).to(self.device)
        mask_hier = 1 - mask_hier[:, None, :]

        unpadded = torch.stack(
            [torch.cat([p, torch.zeros(max_l - p.size(0), src_features.size(-1)).to(self.device)]) for p in unpadded], 1)
        return unpadded, mask_hier


class TransformerInterLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerInterLayer, self).__init__()
        self.d_model, self.heads = d_model, heads
        self.d_per_head = self.d_model // self.heads
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)

        self.layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)

        self.inter_att = MultiHeadedAttention(1, self.d_per_head, dropout, use_final_linear=False)

        self.linear = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask_local, mask_inter, batch_size, n_blocks):
        word_vec = self.layer_norm1(inputs)
        mask_inter = mask_inter.unsqueeze(1).expand(batch_size, self.heads, n_blocks).contiguous()
        mask_inter = mask_inter.view(batch_size * self.heads, 1, n_blocks)

        # block_vec = self.pooling(word_vec, mask_local)

        block_vec = self.pooling(word_vec, word_vec, mask_local)
        block_vec = block_vec.view(-1, self.d_per_head)
        block_vec = self.layer_norm2(block_vec)
        block_vec = block_vec.view(batch_size, n_blocks, self.heads, self.d_per_head)
        block_vec = block_vec.transpose(1, 2).contiguous().view(batch_size * self.heads, n_blocks, self.d_per_head)

        block_vec = self.inter_att(block_vec, block_vec, block_vec, mask_inter)  # all_sents * max_tokens * dim
        block_vec = block_vec.view(batch_size, self.heads, n_blocks, self.d_per_head)
        block_vec = block_vec.transpose(1, 2).contiguous().view(batch_size * n_blocks, self.heads * self.d_per_head)
        block_vec = self.linear(block_vec)

        block_vec = self.dropout(block_vec)
        block_vec = block_vec.view(batch_size * n_blocks, 1, -1)
        out = self.feed_forward(inputs + block_vec)

        return out


class TransformerNewInterLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerNewInterLayer, self).__init__()

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout)

        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.inter_att = MultiHeadedAttention(heads, d_model, dropout, use_final_linear=True)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask_local, mask_inter, batch_size, n_blocks):
        word_vec = self.layer_norm1(inputs)
        mask_inter = mask_inter.unsqueeze(1)
        # block_vec = self.pooling(word_vec, mask_local)

        block_vec = self.pooling(word_vec, word_vec, mask_local)
        _mask_local = ((1 - mask_local).unsqueeze(-1)).float()
        block_vec_avg = torch.sum(word_vec * _mask_local, 1) / (torch.sum(_mask_local, 1) + 1e-9)
        block_vec = self.dropout(block_vec) + block_vec_avg
        block_vec = self.layer_norm2(block_vec)
        block_vec = block_vec.view(batch_size, n_blocks, -1)
        block_vec = self.inter_att(block_vec, block_vec, block_vec, mask_inter)  # all_sents * max_tokens * dim
        block_vec = self.dropout(block_vec)
        block_vec = block_vec.view(batch_size * n_blocks, 1, -1)
        out = self.feed_forward(inputs + block_vec)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)
        self.transformer_local = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_hier = 1 - src.data.eq(padding_idx)
        out = self.pos_emb(emb)

        for i in range(self.num_layers):
            out = self.transformer_local[i](out, out, 1 - mask_hier)  # all_sents * max_tokens * dim
        out = self.layer_norm(out)

        mask_hier = mask_hier[:, :, None].float()
        src_features = out * mask_hier
        src_features = src_features.transpose(0, 1).contiguous()
        mask_hier = mask_hier.transpose(0, 1).contiguous()
        # bridge_feature = self._bridge(src_features, mask_hier)

        # return bridge_feature, src_features, mask_hier
        return src_features, mask_hier
