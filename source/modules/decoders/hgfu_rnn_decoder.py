#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/decoders/hgfu_rnn_decoder.py
"""

import torch
import torch.nn as nn

from source.modules.attention import Attention
from source.modules.decoders.state import DecoderState

from source.utils.misc import Pack
from source.utils.misc import sequence_mask


class RNNDecoder(nn.Module):
    """
    A HGFU GRU recurrent neural network decoder.
    Paper <<Towards Implicit Content-Introducing for Generative Short-Text
            Conversation Systems>>
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 attn_hidden_size=None,
                 memory_size=None,
                 feature_size=None,
                 dropout=0.0,
                 concat=False,
                 with_label=False):
        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.attn_hidden_size = attn_hidden_size or hidden_size // 2
        self.memory_size = memory_size or hidden_size
        self.feature_size = feature_size
        self.dropout = dropout
        self.concat = concat
        self.with_label = with_label

        self.rnn_input_size = self.input_size
        self.out_input_size = self.hidden_size
        self.cue_input_size = self.hidden_size

        if self.feature_size is not None:
            self.rnn_input_size += self.feature_size
            self.cue_input_size += self.feature_size

        if self.attn_mode is not None:
            self.attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.attn_hidden_size,
                                       mode=self.attn_mode,
                                       project=False)
            self.per_word_attention = Attention(query_size=self.hidden_size,
                                                memory_size=self.memory_size,
                                                hidden_size=self.attn_hidden_size,
                                                mode=self.attn_mode,
                                                project=False)
            self.rnn_input_size += self.memory_size
            # self.rnn_input_size += self.hidden_size
            self.cue_input_size += self.memory_size
            self.out_input_size += self.memory_size


        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        self.cue_rnn = nn.GRU(input_size=self.cue_input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=self.dropout if self.num_layers > 1 else 0,
                              batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        # self.fc1_copy = nn.Linear(self.hidden_size, 1)
        # self.fc2_copy = nn.Linear(self.hidden_size, 1)
        # self.fc3_copy = nn.Linear(self.input_size, 1)

        if self.concat:
            self.fc3 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.fc3 = nn.Linear(self.hidden_size * 2, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if self.out_input_size > self.hidden_size:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.hidden_size),
                nn.Linear(self.hidden_size, self.output_size),
                nn.LogSoftmax(dim=-1),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(self.out_input_size, self.output_size),
                nn.LogSoftmax(dim=-1),
            )

    def initialize_state(self,
                         hidden,
                         feature=None,
                         attn_memory=None,
                         attn_mask=None,
                         memory_lengths=None,
                         cue_attn_mask=None,
                         cue_lengths=None,
                         cue_enc_outputs=None,
                         task_id = 1,
                         ):
        """
        initialize_state
        """
        if self.feature_size is not None:
            assert feature is not None

        if self.attn_mode is not None:
            assert attn_memory is not None

        if memory_lengths is not None and attn_mask is None:
            if task_id == 1:
                max_len = attn_memory.size(1)  # 第二阶段attn_memory(batch_size, src_len, 2*rnn_hidden_size)
            else:
                max_len = attn_memory.size(2)  # 第一阶段attn_memory(batch_size, sent_num, num_enc_inputs, 2*rnn_hidden_size)
            attn_mask = sequence_mask(memory_lengths, max_len).eq(0)  # 第一阶段attn_mask(batch_size, sent_num, num_enc_inputs) 第二阶段attn_mask(batch_size, num_enc_inputs)

        if cue_lengths is not None and cue_attn_mask is None:
            cue_max_len = cue_enc_outputs.size(1)  # cue_enc_outputs(batch_size, cue_len, 2*rnn_hidden_size)
            cue_attn_mask = sequence_mask(cue_lengths, cue_max_len).eq(0)  # cue_attn_mask(batch_size,max_len-2)

        init_state = DecoderState(
            hidden=hidden,
            feature=feature,
            attn_memory=attn_memory,
            attn_mask=attn_mask,
            cue_attn_mask=cue_attn_mask,
            cue_enc_outputs=cue_enc_outputs,
            cue_lengths=cue_lengths,
            memory_lengths=memory_lengths,
            task_id = task_id
        )
        return init_state

    def decode(self, input, state, is_training=False):  # 这里是每一个时间步执行一次，注意这里batch_size特指有效长度，即当前时间步无padding的样本数
        """
        decode
        """
        # hidden: src_outputs:(层数， batch_size,  2 * rnn_hidden_size)
        hidden = state.hidden
        task_id = state.task_id
        rnn_input_list = []
        cue_input_list = []
        out_input_list = []  # 为decoder的输出层做准备
        output = Pack()

        if self.embedder is not None:
            input = self.embedder(input)  # (batch_size,input_size)

        input = input.unsqueeze(1)  # (batch_size , 1 , input_size)
        rnn_input_list.append(input)
        # persona = state.cue_enc_outputs  # persona：(batch_size, 1 , 2*rnn_hidden_size)这里的persona是加权和后的persona上下文

        if self.feature_size is not None:
            feature = state.feature.unsqueeze(1)
            rnn_input_list.append(feature)
            cue_input_list.append(feature)

        # 对enc_hidden作attention
        if self.attn_mode is not None:
            # 第二阶段
            if task_id ==1:
                attn_memory = state.attn_memory  # (batch_size , max_len-2, 2*rnn_hidden_size)
                attn_mask = state.attn_mask
                query = hidden[-1].unsqueeze(1)  # (batch_size, 1, 2*rnn_hidden_size)
                weighted_context, attn = self.attention(query=query,
                                                        memory=attn_memory,
                                                        mask=attn_mask)  #attn_mask(batch_size, num_enc_inputs)  weighted_context(batch_size,1, 2*rnn_hidden_size)

            # 第一阶段
            elif task_id == 0:
                ''' 分别对3个相似query做attention'''
                attn_memory = state.attn_memory  # (batch_size,sent_num , max_len-2, 2*rnn_hidden_size)
                batch_size, sent_num, sent_len = attn_memory.size(0), attn_memory.size(1), attn_memory.size(2)
                attn_memory = attn_memory.view(batch_size * sent_num, sent_len, -1)  # (batch_size*sent_num , max_len-2, 2*rnn_hidden_size)
                attn_mask = state.attn_mask.view(batch_size * sent_num, -1)  # attn_mask(batch_size*sent_num, max_len-2) 填充的0全部变成1，其他的变成0
                query = hidden[-1].unsqueeze(1).repeat(1,sent_num,1).view(batch_size*sent_num,1,-1)   # (batch_size*sent_num , 1, 2*rnn_hidden_size)
                weighted_context, attn = self.attention(query=query,
                                                        memory=attn_memory,
                                                        mask=attn_mask)  # weighted_context(batch_size*sent_num, 1 , 2*rnn_hidden_size)
                weighted_context = torch.mean(weighted_context.squeeze(1).view(batch_size, sent_num, -1), dim=1).unsqueeze(1)# weighted_context(batch_size, 1，2*rnn_hidden_size)

            rnn_input_list.append(weighted_context)
            cue_input_list.append(weighted_context)
            out_input_list.append(weighted_context)
            output.add(attn=attn)

            ''' 对persona做attention'''
            cue_attn_memory = state.cue_enc_outputs  # (batch_size, max_len-2, 2*rnn_hidden_size)
            cue_attn_mask = state.cue_attn_mask  # (batch_size,max_len-2)
            cue_query = hidden[-1].unsqueeze(1)  # (batch_size, 1, 2*rnn_hidden_size)
            cue_weighted_context, cue_attn = self.per_word_attention(query=cue_query,
                                                                     memory=cue_attn_memory,
                                                                     mask=cue_attn_mask)
            # cue_weighted_context(batch_size, 1, 2*rnn_hidden_size)
            # cue_attn((batch_size, 1, memory_size))
            cue_input_list.append(cue_weighted_context)
            # out_input_list.append(cue_weighted_context)
            output.add(cue_attn=cue_attn)

        rnn_input = torch.cat(rnn_input_list, dim=-1)  # rnn_input(batch_size, 1 , input_size + 2*rnn_hidden_size + 2*rnn_hidden_size)
        rnn_output, rnn_hidden = self.rnn(rnn_input, hidden)  # rnn_hidden(层数, batch_size , 2*rnn_hidden_size)


        cue_input = torch.cat(cue_input_list, dim=-1)#(batch_size, 1 , 4*rnn_hidden_size)
        cue_output, cue_hidden = self.cue_rnn(cue_input, hidden)#cue_hidden(1, batch_size , 2*rnn_hidden_size)

        h_y = self.tanh(self.fc1(rnn_hidden))
        h_cue = self.tanh(self.fc2(cue_hidden))
        if self.concat:
            new_hidden = self.fc3(torch.cat([h_y, h_cue], dim=-1))#(1, batch_size , 2*rnn_hidden_size)
        else:
            k = self.sigmoid(self.fc3(torch.cat([h_y, h_cue], dim=-1)))
            new_hidden = k * h_y + (1 - k) * h_cue
        state.hidden = new_hidden  # (层数, batch_size , 2*rnn_hidden_size)为下一个时间步更新hidden

        out_input_list.append(new_hidden[-1].unsqueeze(1))  # (batch_size, 1 , 2*rnn_hidden_size)
        out_input = torch.cat(out_input_list, dim=-1)  # (batch_size, 1 , 4*rnn_hidden_size)这里是要输入给为decoder的输出层的，相当于c+h

        if is_training:
            return out_input, state, output  # out_input： 要输入给为decoder的输出层；  state：decoder隐层状态;  output:一个pack字典，包含key"attn"
        else:  # 一个时间步           #out_input(batch_size, 1 , 4*rnn_hidden_size)这里是要输入给为decoder的输出层的，相当于c+h
            log_prob = self.output_layer(out_input)
            return log_prob, state, output

    def forward(self, inputs, state):
        """
        forward
        """
        # if self.with_label:
        inputs, lengths = inputs  # inputs:(batch_size，max_len-1)“tgt”去尾没有eos符号
        batch_size, max_len = inputs.size()

        out_inputs = inputs.new_zeros(size=(batch_size, max_len, self.out_input_size), dtype=torch.float)# （batch_size, max_len, 4*rnn_hidden_size）

        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)  # 更改en_hidden的batch顺序，使其与tgt长度排序顺序相同

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)  # (max_len-1)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)
            out_input, valid_state, _ = self.decode(dec_input, valid_state, is_training=True)
            state.hidden[:, :num_valid] = valid_state.hidden
            out_inputs[:num_valid, i] = out_input.squeeze(1)  # (num_valid, max_len, 4*rnn_hidden_size）

        # Resort 撤回排序
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)

        log_probs = self.output_layer(out_inputs)  # softmax输出置信度:(batch_size, max_len, tgt_vacb_size)
        return log_probs, state