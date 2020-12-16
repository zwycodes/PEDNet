#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/utils/misc.py
"""

import torch
import argparse
# argparse 是python自带的命令行参数解析包，可以用来方便地读取命令行参数


class Pack(dict):
    """
    Pack
    """
    def __getattr__(self, name):#__getattr__为内置方法，当使用点号获取实例属性时，如果属性不存在就自动调用__getattr__方法
        return self.get(name) #get函数返回指定键name的值，如果值不在字典中返回默认值None

    def add(self, **kwargs):
        """
        add
        """
        for k, v in kwargs.items():
            self[k] = v

    def flatten(self):
        """
        flatten
        """
        pack_list = [] #dict.values()以列表形式返回字典中的所有值，zip(*zipped) 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
        for vs in zip(*self.values()): #dict={'A':(1, 4),'B':(2, 5),'C': (3, 6)} 那么dict.values=[(1, 4), (2, 5), (3, 6)],然后解压为[(1, 2, 3), (4, 5, 6)]
            pack = Pack(zip(self.keys(), vs))  #打包成元组组成的列表,如第一次循环，zip(['A','B','C'],(1,2,3)) 得到[('A',1),('B',2),('C',3)],第二次[('A',4),('B',5),('C',6)]
            pack_list.append(pack)
        return pack_list

    def cuda(self, device=None):
        """
        cuda
        """
        pack = Pack()
        for k, v in self.items():
            if isinstance(v, tuple):
                pack[k] = tuple(x.cuda(device) for x in v)
            else:
                pack[k] = v.cuda(device)
        return pack


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    if max_len is None:
        max_len = lengths.max().item() #lengths是一个tensor，max函数取出最大的元素，item()取出数值
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)#(1 , max_len)
    mask = mask.repeat(1, *lengths.size(), 1) #注意这里的*号
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    #mask = mask.repeat(*lengths.size(), 1).lt(lengths.unsqueeze(-1))
    return mask#(batch_size , max_len)


def max_lens(X):
    """
    max_lens
    """
    if not isinstance(X[0], list):
        return [len(X)] #第一个元素不是列表的话，返回X的长度
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X),
                max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")


def list2tensor(X):#将list转换为tensor，length用来sequence的长度
    """
    list2tensor
    """
    size = max_lens(X) #size是一个列表
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)#填充长度不够的为0
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:
        for i, x in enumerate(X):
            l = len(x)
            tensor[i, :l] = torch.tensor(x)
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                l = len(x)
                tensor[i, j, :l] = torch.tensor(x)
                lengths[i, j] = l

    return tensor, lengths


# def list2tensor(X, max_len=None):
#     sizes = max_lens(X)
#
#     if len(sizes) == 1:
#         tensor = torch.tensor(X)
#         return tensor
#
#     if max_len is not None:
#         assert max_len >= sizes[-1]
#         sizes[-1] = max_len
#
#     tensor = torch.zeros(sizes, dtype=torch.long)
#     lengths = torch.zeros(sizes[:-1], dtype=torch.long)
#     if len(sizes) == 2:
#         for i, x in enumerate(X):
#             l = len(x)
#             tensor[i, :l] = torch.tensor(x)
#             lengths[i] = l
#     else:
#         for i, xs in enumerate(X):
#             for j, x in enumerate(xs):
#                 l = len(x)
#                 tensor[i, j, :l] = torch.tensor(x)
#                 lengths[i, j] = l
#
#     return tensor, lengths


# def one_hot(indice, vocab_size):
#     T = torch.zeros(*indice.size(), vocab_size).type_as(indice).float()
#     T = T.scatter(-1, indice.unsqueeze(-1), 1)
#     return T

def one_hot(indice, num_classes):
    """
    one_hot
    """
    I = torch.eye(num_classes).to(indice.device) #device =x.device, y.to(device)
    T = I[indice]
    return T


def str2bool(v):
    """
    str2bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    X = [1, 2, 3]
    print(X)
    print(list2tensor(X))
    X = [X, [2, 3]]
    print(X)
    print(list2tensor(X))
    X = [X, [[1, 1, 1, 1, 1]]]
    print(X)
    print(list2tensor(X))

    data_list = [{'src': [1, 2, 3], 'tgt': [1, 2, 3, 4]},
                 {'src': [2, 3], 'tgt': [1, 2, 4]}]
    batch = Pack()
    for key in data_list[0].keys():
        batch[key] = list2tensor([x[key] for x in data_list], 8)
    print(batch)
