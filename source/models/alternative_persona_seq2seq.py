#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/models/idf_persona_seq2seq.py
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder import RNNEncoder
from source.modules.decoders.hgfu_rnn_decoder import RNNDecoder
from source.utils.criterions import NLLLoss
from source.utils.misc import Pack
from source.utils.metrics import accuracy
from source.utils.metrics import attn_accuracy
from source.utils.metrics import perplexity
from source.utils.misc import sequence_mask
from source.modules.attention import Attention


class TwoStagePersonaSeq2Seq(BaseModel):
    """
    TwoStagePersonaSeq2Seq
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, padding_idx=None,
                 num_layers=1, bidirectional=True, attn_mode="mlp", attn_hidden_size=None,
                 with_bridge=False, tie_embedding=False, dropout=0.0, use_gpu=False, use_dssm=False,
                 weight_control=False, use_pg=False, concat=False, pretrain_epoch=0, with_label=False):
        super(TwoStagePersonaSeq2Seq, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size  
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.attn_hidden_size = attn_hidden_size
        self.with_bridge = with_bridge
        self.tie_embedding = tie_embedding
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.use_dssm = use_dssm
        self.weight_control = weight_control
        self.use_pg = use_pg
        self.pretrain_epoch = pretrain_epoch
        self.baseline = 0
        self.with_label = with_label
        self.task_id = 1

        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size, padding_idx=self.padding_idx)

        self.encoder = RNNEncoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                  embedder=enc_embedder, num_layers=self.num_layers,
                                  bidirectional=self.bidirectional, dropout=self.dropout)

        if self.with_bridge:
            self.bridge = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())

        if self.tie_embedding: 
            assert self.src_vocab_size == self.tgt_vocab_size
            dec_embedder = enc_embedder
            persona_embedder = enc_embedder
        else:
            dec_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                    embedding_dim=self.embed_size, padding_idx=self.padding_idx)
            persona_embedder = Embedder(num_embeddings=self.tgt_vocab_size,
                                        embedding_dim=self.embed_size,
                                        padding_idx=self.padding_idx)

        self.persona_encoder = RNNEncoder(input_size=self.embed_size,
                                          hidden_size=self.hidden_size,
                                          embedder=persona_embedder,
                                          num_layers=self.num_layers,
                                          bidirectional=self.bidirectional,
                                          dropout=self.dropout)

        self.persona_attention = Attention(query_size=self.hidden_size,
                                           memory_size=self.hidden_size,
                                           hidden_size=self.hidden_size,
                                           mode="general")

        self.decoder = RNNDecoder(input_size=self.embed_size, hidden_size=self.hidden_size,
                                  output_size=self.tgt_vocab_size, embedder=dec_embedder,
                                  num_layers=self.num_layers, attn_mode=self.attn_mode,
                                  memory_size=self.hidden_size, feature_size=None,
                                  dropout=self.dropout, concat=concat, with_label=self.with_label)
        self.key_linear = nn.Linear(in_features=self.embed_size,
                                    out_features=self.hidden_size)

        if self.use_dssm:
            self.dssm_project = nn.Linear(in_features=self.hidden_size,
                                          out_features=self.hidden_size)
            self.mse_loss = torch.nn.MSELoss(reduction='mean')

        self.mse_loss = torch.nn.MSELoss(reduction='mean')

        if self.padding_idx is not None:
            self.weight = torch.ones(self.tgt_vocab_size)
            self.weight[self.padding_idx] = 0
        else:
            self.weight = None
        self.nll_loss = NLLLoss(weight=self.weight, ignore_index=self.padding_idx,
                                reduction='mean')

        self.persona_loss = NLLLoss(weight=None, reduction='mean')
        self.eps = 1e-7

        if self.use_gpu:
            self.cuda()
            self.weight = self.weight.cuda()

    def encode(self, inputs, hidden=None, is_training=False):
        """
        encode
        """
        '''
	    #inputs: 嵌套形式为{分离src和target和cue->(分离数据和长度->tensor数据值    
	    #{'src':( 数据值-->shape(batch_size , sen_num , max_len), 句子长度值--> shape(batch_size，sen_num) ),
          'tgt':( 数据值-->shape(batch_size , max_len), 句子长度值-->shape(batch_size) )，
          'cue' :( 数据值-->shape(batch_size, max_len), 句子长度值--> shape(batch_size) ),
          'label':( 数据值-->shape(batch_size , max_len), 句子长度值-->shape(batch_size) )，
          'index': ( 数据值-->shape(batch_size , max_len), 句子长度值-->shape(batch_size) )
          }
	    '''
        outputs = Pack()  
        ''' 第二阶段'''
        if self.task_id==1:
    
            enc_inputs = inputs.src[0][:, 1:-1], inputs.src[1] - 2
            lengths = inputs.src[1] - 2  # (batch_size)
            enc_outputs, enc_hidden, enc_embedding = self.encoder(enc_inputs, hidden)
            # enc_outputs:(batch_size, max_len-2, 2*rnn_hidden_size)
            # enc_hidden:(num_layer , batch_size , 2*rnn_hidden_size)

            if self.with_bridge:
                enc_hidden = self.bridge(enc_hidden)

           
            # tem_bth,tem_len,tem_hi_size =enc_outputs.size()# batch_size, max_len-2, 2*rnn_hidden_size)
            key_index, len_key_index = inputs.index[0], inputs.index[1]  # key_index(batch_size , idx_max_len)
            max_len = key_index.size(1)
            key_mask = sequence_mask(len_key_index, max_len).eq(0)  # key_mask(batch_size , idx_max_len)
            key_hidden = torch.gather( enc_embedding, 1, key_index.unsqueeze(-1).repeat(1, 1, enc_embedding.size(-1)))  # (batch_size ,idx_max_len, 2*rnn_hidden_size)
            key_global = key_hidden.masked_fill(key_mask.unsqueeze(-1),0.0).sum(1) / len_key_index.unsqueeze(1).float()
            key_global = self.key_linear(key_global) # (batch_size, 2*rnn_hidden_size)
            # persona_aware = torch.cat([key_global, enc_hidden[-1]], dim=-1)  # (batch_size ,2*rnn_hidden_size)
            persona_aware = key_global + enc_hidden[-1]#(batch_size , 2*rnn_hidden_size)

            # persona
            batch_size, sent_num, sent = inputs.cue[0].size()
            cue_len = inputs.cue[1]  # (batch_size，sen_num)
            cue_len[cue_len > 0] -= 2  # (batch_size, sen_num)
            cue_inputs = inputs.cue[0].view(-1, sent)[:, 1:-1], cue_len.view(-1)
            # cue_inputs:((batch_size*sent_num , max_len-2),(batch_size*sent_num))
            cue_enc_outputs, cue_enc_hidden ,_= self.persona_encoder(cue_inputs, hidden)
            # cue_enc_outputs:(batch_size*sent_num , max_len-2, 2*rnn_hidden_size)
            # cue_enc_hidden:(层数 , batch_size*sent_num, 2 * rnn_hidden_size)
            cue_outputs = cue_enc_hidden[-1].view(batch_size, sent_num,
                                                  -1)  
            cue_enc_outputs = cue_enc_outputs.view(batch_size, sent_num, cue_enc_outputs.size(1),
                                                   -1)  # cue_enc_outputs:(batch_size, sent_num , max_len-2, 2*rnn_hidden_size)
            cue_len = cue_len.view(batch_size, sent_num)

            # cue_outputs:(batch_size, sent_num, 2 * rnn_hidden_size)
            # Attention
            weighted_cue1, cue_attn1 = self.persona_attention(query=persona_aware.unsqueeze(1),
                                                            memory=cue_outputs,
                                                            mask=inputs.cue[1].eq(0))
            # weighted_cue:(batch_size , 1 , 2 * rnn_hidden_size)
            persona_memory1 = weighted_cue1 + persona_aware.unsqueeze(1)
            weighted_cue2, cue_attn2 = self.persona_attention(query=persona_memory1,
                                                            memory=cue_outputs,
                                                            mask=inputs.cue[1].eq(0))
            persona_memory2 = weighted_cue2 + persona_aware.unsqueeze(1)
            weighted_cue3, cue_attn3 = self.persona_attention(query=persona_memory2,
                                                              memory=cue_outputs,
                                                              mask=inputs.cue[1].eq(0))


            cue_attn = cue_attn3.squeeze(1)
            # cue_attn:(batch_size, sent_num)
            outputs.add(cue_attn=cue_attn)
            indexs = cue_attn.max(dim=1)[1]  # (batch_size)
            if is_training:
                # gumbel_attn = F.gumbel_softmax(torch.log(cue_attn + 1e-10), 0.1, hard=True)
                # persona = torch.bmm(gumbel_attn.unsqueeze(1), cue_outputs)
                # indexs = gumbel_attn.max(-1)[1]
                # cue_lengths = cue_len.gather(1, indexs.unsqueeze(1)).squeeze(1)  # (batch_size)
                persona = cue_enc_outputs.gather(1, indexs.view(-1, 1, 1, 1).repeat(1, 1, cue_enc_outputs.size(2),
                                                                                    cue_enc_outputs.size(3))).squeeze(
                    1)  # (batch_size , max_len-2, 2*rnn_hidden_size)
                cue_lengths = cue_len.gather(1, indexs.unsqueeze(1)).squeeze(1)  # (batch_size)
            else:
                persona = cue_enc_outputs.gather(1, indexs.view(-1, 1, 1, 1).repeat(1, 1, cue_enc_outputs.size(2),
                                                                                    cue_enc_outputs.size(3))).squeeze(
                    1)  # (batch_size , max_len-2, 2*rnn_hidden_size)
                cue_lengths = cue_len.gather(1, indexs.unsqueeze(1)).squeeze(1)  # (batch_size)


            outputs.add(indexs=indexs)
            outputs.add(attn_index=inputs.label)  # (batch_size)

            dec_init_state = self.decoder.initialize_state(
                hidden=enc_hidden,
                attn_memory=enc_outputs if self.attn_mode else None,
                memory_lengths=lengths if self.attn_mode else None,  # (batch_size)
                cue_enc_outputs=persona,  # (batch_size, max_len-2, 2*rnn_hidden_size)
                cue_lengths=cue_lengths,  # (batch_size)
                task_id = self.task_id
            )

            # if 'index' in inputs.keys():
            #     outputs.add(attn_index=inputs.index)


        elif self.task_id==0:
            ''' 第一阶段'''
            # enc_inputs:((batch_size，max_len-2), (batch_size-2))**src去头去尾
            # hidden:None
            batch_size, sent_num, sent_len = inputs.src[0].size()
            src_lengths = inputs.src[1]  # (batch_size，sent_num)
            src_lengths[src_lengths > 0] -= 2
            # src_lengths(batch_size, sent_num)
            src_inputs = inputs.src[0].view(-1, sent_len)[:, 1:-1], src_lengths.view(-1)
            # src_inputs:((batch_size*sent_num , max_len-2),(batch_size*sent_num))
            src_enc_outputs, enc_hidden,_ = self.encoder(src_inputs, hidden)

            if self.with_bridge:
                enc_hidden = self.bridge(enc_hidden)

            # src_enc_outputs:(batch_size*sent_num , max_len-2, 2*rnn_hidden_size)
            # enc_hidden:(层数 , batch_size*sent_num, 2 * rnn_hidden_size)
            src_outputs = torch.mean(enc_hidden.view(self.num_layers, batch_size, sent_num, -1), 2)  # 池化
            # src_outputs:(层数，batch_size,  2 * rnn_hidden_size)

            # persona:((batch_size，max_len-2), (batch_size))**persona的Tensor去头去尾
            cue_inputs = inputs.cue[0][:, 1:-1], inputs.cue[1] - 2
            cue_lengths = inputs.cue[1] - 2  # (batch_size)
            cue_enc_outputs, cue_enc_hidden,_ = self.persona_encoder(cue_inputs, hidden)
            # cue_enc_outputs:(batch_size, max_len-2, 2*rnn_hidden_size)
            # cue_enc_hidden:(num_layer , batch_size , 2*rnn_hidden_size)

            dec_init_state = self.decoder.initialize_state(
                hidden=src_outputs,  
                attn_memory=src_enc_outputs.view(batch_size, sent_num, sent_len - 2, -1) if self.attn_mode else None,
                # (batch_size, sent_num , max_len-2, 2*rnn_hidden_size)
                memory_lengths=src_lengths if self.attn_mode else None,  # (batch_size，sent_num)
                cue_enc_outputs=cue_enc_outputs,  # (batch_size, max_len-2, 2*rnn_hidden_size)
                cue_lengths=cue_lengths,
                task_id = self.task_id  # (batch_size)
            )
        return outputs, dec_init_state  

    def decode(self, input, state):

        """
        decode
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

    def forward(self, enc_inputs, dec_inputs, hidden=None, is_training=False):
        """
        forward
        """
        outputs, dec_init_state = self.encode(
            enc_inputs, hidden, is_training=is_training)
        log_probs, _ = self.decoder(dec_inputs,
                                    dec_init_state)  
        outputs.add(logits=log_probs)
        return outputs

    def collect_metrics(self, outputs, target, epoch=-1):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        # test begin
        # nll = self.nll(torch.log(outputs.posterior_attn+1e-10), outputs.attn_index)
        # loss += nll
        # attn_acc = attn_accuracy(outputs.posterior_attn, outputs.attn_index)
        # metrics.add(attn_acc=attn_acc)
        # metrics.add(loss=loss)
        # return metrics
        # test end

        logits = outputs.logits
        scores = -self.nll_loss(logits, target, reduction=False)
        nll_loss = self.nll_loss(logits, target)
        num_words = target.ne(self.padding_idx).sum().item()
        acc = accuracy(logits, target, padding_idx=self.padding_idx)
        metrics.add(nll=(nll_loss, num_words), acc=acc)
        # persona loss
        if 'attn_index' in outputs:
            attn_acc = attn_accuracy(outputs.cue_attn, outputs.attn_index)
            metrics.add(attn_acc=attn_acc)
            per_logits = torch.log(outputs.cue_attn + self.eps)  # cue_attn(batch_size, sent_num)
            per_labels = outputs.attn_index  ##(batch_size)
            use_per_loss = self.persona_loss(per_logits, per_labels)  # per_labels(batch_size)
            metrics.add(use_per_loss=use_per_loss)
            loss += 0.7*use_per_loss
            loss += 0.3*nll_loss
        else:
            loss += nll_loss

        metrics.add(loss=loss)
        return metrics, scores

    def iterate(self, inputs, optimizer=None, grad_clip=None, is_training=False, epoch=-1, task_id = 1):
        """
        iterate
        """
        
        if is_training:
            self.task_id = task_id
            enc_inputs = inputs[task_id]
            dec_inputs = inputs[task_id].tgt[0][:, :-1], inputs[task_id].tgt[1] - 1  
            target = inputs[task_id].tgt[0][:, 1:] 
        else:
            self.task_id = 1
            enc_inputs = inputs[1]
            dec_inputs = inputs[1].tgt[0][:, :-1], inputs[1].tgt[1] - 1 
            target = inputs[1].tgt[0][:, 1:]  

        outputs = self.forward(enc_inputs, dec_inputs, is_training=is_training)
        metrics, scores = self.collect_metrics(outputs, target, epoch=epoch)

        loss = metrics.loss
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        if is_training:
            if self.use_pg:
                self.baseline = 0.99 * self.baseline + 0.01 * metrics.reward.item()
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()
        return metrics, scores
