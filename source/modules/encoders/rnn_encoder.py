#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/encoders/rnn_encoder.py
"""

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class RNNEncoder(nn.Module):
    """
    A GRU recurrent neural network encoder.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 embedder=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        rnn_hidden_size = hidden_size // num_directions #双斜杠（//）表示地板除，即先做除法，然后向下取整。至少有一方是float型时，结果为float型；两个数都是int型时，结果为int型。



        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.num_directions=num_directions
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          bidirectional=self.bidirectional)

    def forward(self, inputs, hidden=None):
        """
        forward
        """
        #inputs:(batch_size, max_len-2)**src去头去尾
        #lengths:(batch_size)
        #hidden:None
        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None

        if self.embedder is not None:
            rnn_inputs = self.embedder(inputs)
            enc_embedding = rnn_inputs
        else:
            rnn_inputs = inputs
        #rnn_inputs:(batch_size , max_len-2 , embedding_dim)
        batch_size = rnn_inputs.size(0)

        if lengths is not None:#根据长度重排序,并去除无效长度的样本（样本是指一个batch中的一条数据）
            num_valid = lengths.gt(0).int().sum().item()
            sorted_lengths, indices = lengths.sort(descending=True)
            rnn_inputs = rnn_inputs.index_select(0, indices)#在第0维选择indices的序列，也就是根据长度对序列重新排序了

            rnn_inputs = pack_padded_sequence(
                rnn_inputs[:num_valid],
                sorted_lengths[:num_valid].tolist(),
                batch_first=True)

            if hidden is not None:
                hidden = hidden.index_select(1, indices)[:, :num_valid]

        outputs, last_hidden = self.rnn(rnn_inputs, hidden)
        #outputs:(batch_size, max_len-2, 方向*rnn_hidden_size)
        #last_hidden:(层数*方向 ， batch_size ， rnn_hidden_size)
        if self.bidirectional:
            last_hidden = self._bridge_bidirectional_hidden(last_hidden)
        #last_hidden:(层数, batch_size, 方向*rnn_hidden_size)
        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

            if num_valid < batch_size:#填补无效长度的样本
                zeros = outputs.new_zeros(
                    batch_size - num_valid, outputs.size(1), self.hidden_size)
                outputs = torch.cat([outputs, zeros], dim=0)

                zeros = last_hidden.new_zeros(
                    self.num_layers, batch_size - num_valid, self.hidden_size)
                last_hidden = torch.cat([last_hidden, zeros], dim=1)

            _, inv_indices = indices.sort()
            outputs = outputs.index_select(0, inv_indices)#将上述排序撤回
            last_hidden = last_hidden.index_select(1, inv_indices)

        return outputs, last_hidden, enc_embedding

    def _bridge_bidirectional_hidden(self, hidden):#如果是双向网络，过完了rnn之后，将last_hidden的维度进行变换
        """
        the bidirectional hidden is (num_layers * num_directions, batch_size, hidden_size)
        we need to convert it to (num_layers, batch_size, num_directions * rnn_hidden_size)
        """
        num_layers = hidden.size(0) // self.num_directions
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, self.num_directions, batch_size, hidden_size)\
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * self.num_directions)


class HRNNEncoder(nn.Module):
    """
    HRNNEncoder
    """
    def __init__(self,
                 sub_encoder,
                 hiera_encoder):
        super(HRNNEncoder, self).__init__()
        self.sub_encoder = sub_encoder
        self.hiera_encoder = hiera_encoder

    def forward(self, inputs, features=None, sub_hidden=None, hiera_hidden=None,
                return_last_sub_outputs=False):
        """
        inputs: Tuple[Tensor(batch_size, max_hiera_len, max_sub_len), 
                Tensor(batch_size, max_hiera_len)]
        """
        indices, lengths = inputs
        batch_size, max_hiera_len, max_sub_len = indices.size()
        hiera_lengths = lengths.gt(0).long().sum(dim=1)

        # Forward of sub encoder
        indices = indices.view(-1, max_sub_len)
        sub_lengths = lengths.view(-1)
        sub_enc_inputs = (indices, sub_lengths)
        sub_outputs, sub_hidden = self.sub_encoder(sub_enc_inputs, sub_hidden)
        sub_hidden = sub_hidden[-1].view(batch_size, max_hiera_len, -1)

        if features is not None:
            sub_hidden = torch.cat([sub_hidden, features], dim=-1)

        # Forward of hiera encoder
        hiera_enc_inputs = (sub_hidden, hiera_lengths)
        hiera_outputs, hiera_hidden = self.hiera_encoder(
            hiera_enc_inputs, hiera_hidden)

        if return_last_sub_outputs:
            sub_outputs = sub_outputs.view(
                batch_size, max_hiera_len, max_sub_len, -1)
            last_sub_outputs = torch.stack(
                [sub_outputs[b, l - 1] for b, l in enumerate(hiera_lengths)])
            last_sub_lengths = torch.stack(
                [lengths[b, l - 1] for b, l in enumerate(hiera_lengths)])
            max_len = last_sub_lengths.max()
            last_sub_outputs = last_sub_outputs[:, :max_len]
            return hiera_outputs, hiera_hidden, (last_sub_outputs, last_sub_lengths)
        else:
            return hiera_outputs, hiera_hidden, None
