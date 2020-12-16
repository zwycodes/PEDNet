#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/dataset.py
"""

import torch
from torch.utils.data import DataLoader

from source.utils.misc import Pack
from source.utils.misc import list2tensor


class Dataset(torch.utils.data.Dataset):
    """
    Dataset
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):#定义当被len()函数调用时的行为（返回容器中元素的个数）
        return len(self.data[1])

    def __getitem__(self, idx):#定义获取容器中指定元素的行为，相当于self[key]，即允许类对象可以有索引操作
        return (self.data[0][idx],self.data[1][idx])

    @staticmethod
    def collate_fn(device=-1):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            """
            data_list1,data_list2 = zip(*data_list)
            batch1 = Pack()
            batch2 = Pack()
            data_list1 = list(data_list1)
            data_list2 = list(data_list2)
            for key in data_list1[0].keys():
                batch1[key] = list2tensor([x[key] for x in data_list1])
            if device >= 0:
                batch1 = batch1.cuda(device=device)
            for key in list(data_list2)[0].keys():
                batch2[key] = list2tensor([x[key] for x in data_list2])
            if device >= 0:
                batch2 = batch2.cuda(device=device)
            return batch1,batch2
        return collate

    def create_batches(self, batch_size=1, shuffle=False, device=-1):
        """
        create_batches
        """
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        return loader
