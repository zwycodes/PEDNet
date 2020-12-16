#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/corpus.py
"""

import os
import torch

from tqdm import tqdm
from source.inputters.field import tokenize
from source.inputters.field import TextField
from source.inputters.field import NumberField
from source.inputters.dataset import Dataset


class Corpus(object):
    """
    Corpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size#max_vocab_size是在建词表时给定的最大词表长度

        prepared_data_file = data_prefix + "_" + str(max_vocab_size) + ".data.pt"
        prepared_vocab_file = data_prefix + "_" + str(max_vocab_size) + ".vocab.pt"

        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(data_dir, prepared_vocab_file)
        self.fields = {}
        self.filter_pred = None
        self.sort_fn = None
        self.data = None

    def load(self):#加载准备文件
        """
        load
        """
        if not (os.path.exists(self.prepared_data_file) or os.path.exists(self.prepared_vocab_file)):
            self.build()#都不存在，执行build()方法
        elif os.path.exists(self.prepared_vocab_file) and not os.path.exists(self.prepared_data_file):
            self.load_vocab(self.prepared_vocab_file)
            self.build()
        else:
            self.load_vocab(self.prepared_vocab_file)
            self.load_data(self.prepared_data_file)#在这里将数据变成Dataset

        self.padding_idx = self.TGT.stoi[self.TGT.pad_token]#padding_idx=0

    def reload(self, data_type='test'):
        """
        reload
        """
        data_file1 = os.path.join(self.data_dir, self.data_prefix + "." + 'test1')
        data_file2 = os.path.join(self.data_dir, self.data_prefix + "." + 'test2')
        data_raw1,data_raw2 = self.read_data_multitask(data_file1, data_file2,data_type="test")
        data_examples1 = self.build_examples(data_raw1)
        data_examples2 = self.build_examples(data_raw2)
        self.data[data_type] = Dataset((data_examples1,data_examples2))

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        load_data,读取处理好的语料数据
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        print(len(data['train']))
        print(len(data['valid']))
        print(len(data['valid']))
    
        self.data = {"train": Dataset(data['train']),
                     "valid": Dataset(data["valid"]),
                     "test": Dataset(data["test"])}

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_vocab(self, prepared_vocab_file):
        """
        load_vocab，读取处理好的词表数据
        """
        prepared_vocab_file = prepared_vocab_file or self.prepared_vocab_file
        print("Loading prepared vocab from {} ...".format(prepared_vocab_file))
        vocab_dict = torch.load(prepared_vocab_file)

        for name, vocab in vocab_dict.items():
            if name in self.fields:#self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE}
                self.fields[name].load_vocab(vocab)
        print("Vocabulary size of fields:",
              " ".join("{}-{}".format(name.upper(), field.vocab_size) 
                for name, field in self.fields.items() 
                    if isinstance(field, TextField)))

    def read_data(self, data_file, data_type=None):
        """
        Returns
        -------
        data: ``List[Dict]``
        """
        raise NotImplementedError

    def build_vocab(self, data):
        """
        Args
        ----
        """
        #第一阶段data=[{'src':['query1','query2','query3'], 'tgt': 'response', 'cue':'persona1'},{...},...]
        #第二阶段data=[{'src':'query', 'tgt': 'response', 'cue':['persona1','persona2',...],'label':'2','index':'14 15 17'},{...},...]
        #第一阶段self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE}
        #第二阶段self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE, 'label': self.LABEL, 'index': self.INDEX}
        field_data_dict = {}
        for name in data[0].keys():#name='src'/'tgt'/'cue'  keys返回的是一个迭代器
            field = self.fields.get(name)#field=self.SRC/self.TGT/self.CUE 
            if isinstance(field, TextField):#只对文本的TextField做处理，NumberField不做处理
                xs = [x[name] for x in data]#所有数据的src/tgt/cue    #x['src']/x['tgt']/x['cue']
                if field not in field_data_dict:#不分享词表
                    field_data_dict[field] = xs
                else:#分享词表
                    field_data_dict[field] += xs
        '''
        不分享词表：field_data_dict = {self.SRC:[src1,src2,src2,…]，self.TGT:[tgt1,tgt2,tgt3,…], self.CUE:[cue1,cue1,cue3,…]} 
        分享词表：field_data_dict = {self.SRC:[src1,src2,src2,…, tgt1,tgt2,tgt3,…, cue1,cue1,cue3,…]}
        '''
        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)
                vocab_dict[name] = field.dump_vocab()
        return vocab_dict#{'src':词表，'tgt':词表，'cue':词表}  词表：{"itos": self.itos,"embeddings": self.embeddings} 共享词表后这里的词表都是一样的

    def build_examples(self, data):
        """
        Args
        ----
        第一阶段self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE}
        第二阶段self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE, 'label': self.LABEL, 'index': self.INDEX}
        
        这时的data未分词
        """
        
        #第一阶段data=[{'src':['query1','query2','query3'], 'tgt': 'response', 'cue':'persona'},{...},...]
        #第二阶段data=[{'src':'query', 'tgt': 'response', 'cue':['persona1','persona2',...],'label':'2','index':'14 15 17'},{...},...]
        examples = []
        for raw_data in tqdm(data):
            example = {}
            for name, strings in raw_data.items():#strings有可能是一句话的字符串，也有可能是列表（列表元素是persona文本）
                example[name] = self.fields[name].numericalize(strings)#将字符串分词、加上开始符号和结束符号并且索引化，一句话变成一个列表
            examples.append(example) #examples=[{'src':[[2，25,35,66，3],[2，46.56,,3],[2,45,3]], 'tgt':[2，22,45,33,98,101，3],'cue':[2，5,10,36,99，3]},{},{}...]
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples   #examples数据形式（索引，已分词，已加前后缀）

    def build(self):
        """
        build
        """
        print("Start to build corpus!")
        train_file1 = os.path.join(self.data_dir, self.data_prefix + ".train1")
        valid_file1 = os.path.join(self.data_dir, self.data_prefix + ".dev1")
        test_file1 = os.path.join(self.data_dir, self.data_prefix + ".test1")
        train_file2 = os.path.join(self.data_dir, self.data_prefix + ".train2")
        valid_file2 = os.path.join(self.data_dir, self.data_prefix + ".dev2")
        test_file2 = os.path.join(self.data_dir, self.data_prefix + ".test2")

        print("Reading data ...")
        #_raw数据形式（原生文本未分词）：列表嵌套字典
        train_raw1, train_raw2 = self.read_data_multitask(train_file1, train_file2, data_type="train")
        valid_raw1, valid_raw2 = self.read_data_multitask(valid_file1,valid_file2, data_type="valid")
        test_raw1, test_raw2= self.read_data_multitask(test_file1,test_file2, data_type="test")#
        #第一阶段train_raw=[{'src':['query1','query2','query3'], 'tgt': 'response', 'cue':'persona'},{...},...]
        #第二阶段train_raw=[{'src':'query', 'tgt': 'response', 'cue':['persona1','persona2',...],'label':'2','index':'14 15 17'},{...},...]
        # valid_raw = self.read_data(valid_file, data_type="valid")
        # test_raw = self.read_data(test_file, data_type="test")

        #这里建词表
        if not os.path.exists(self.prepared_vocab_file):
            vocab = self.build_vocab(train_raw2)#{"src":{"itos": self.itos,"embeddings": self.embeddings},"tgt":{"itos": self.itos,"embeddings": self.embeddings},"cue":{"itos": self.itos,"embeddings": self.embeddings}} 共享词表后，3个{"itos": self.itos,"embeddings": self.embeddings}一样
            print("Saving prepared vocab ...")
            torch.save(vocab, self.prepared_vocab_file)
            print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))
        
        print("Building TRAIN examples ...")
        train_data1 = self.build_examples(train_raw1) #examples将读取的数据data分词并变成索引
        train_data2 = self.build_examples(train_raw2)
        '''
        train_data=[{'src':[5,10,36,99],[33,21,56],[9,26,66,38,47]'tgt':[22,45,33,98,101],'cue':[25,35,66]},
                    {},
                    {},...]
        '''
        print("Building VALID examples ...")
        valid_data1 = self.build_examples(valid_raw1)
        valid_data2 = self.build_examples(valid_raw2)
        print("Building TEST examples ...")
        test_data1 = self.build_examples(test_raw1)
        test_data2 = self.build_examples(test_raw2)


        data = {"train": (train_data1,train_data2),
                "valid": (valid_data1,valid_data2),
                "test": (test_data1,test_data2)}

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))


    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        """
        try:
            data = self.data[data_type] #data=Dataset(data['train'])/Dataset(data['valid'])/Dataset(data['test'])
            data_loader = data.create_batches(batch_size, shuffle, device) 
            return data_loader#Dataset创建批训练迭代器，每批数据中的每条数据变成(数据的tensor值，句子长度)这样的数据
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader


class SrcTgtCorpus(Corpus):
    """
    SrcTgtCorpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None,
                 min_len=0,
                 max_len=100,
                 embed_file=None,
                 share_vocab=False):
        super(SrcTgtCorpus, self).__init__(data_dir=data_dir,
                                           data_prefix=data_prefix,
                                           min_freq=min_freq,
                                           max_vocab_size=max_vocab_size)
        self.min_len = min_len
        self.max_len = max_len
        self.share_vocab = share_vocab

        self.SRC = TextField(tokenize_fn=tokenize,
                             embed_file=embed_file)
        if self.share_vocab:
            self.TGT = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)

        self.fields = {'src': self.SRC, 'tgt': self.TGT}

        def src_filter_pred(src):
            """
            src_filter_pred
            """
            return min_len <= len(self.SRC.tokenize_fn(src)) <= max_len

        def tgt_filter_pred(tgt):
            """
            tgt_filter_pred
            """
            return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len

        self.filter_pred = lambda ex: src_filter_pred(ex['src']) and tgt_filter_pred(ex['tgt'])

    def read_data(self, data_file, data_type="train"):
        """
        read_data
        """
        data = []
        filtered = 0
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                src, tgt = line.strip().split('\t')[:2]
                data.append({'src': src, 'tgt': tgt})

        filtered_num = len(data)
        if self.filter_pred is not None:
            data = [ex for ex in data if self.filter_pred(ex)]
        filtered_num -= len(data)
        print(
            "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data


class PersonaCorpus(Corpus):
    """
    PersonaCorpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None,
                 min_len=0,
                 max_len=100,
                 embed_file=None,
                 share_vocab=False,
                 with_label=False):
        super(PersonaCorpus, self).__init__(data_dir=data_dir,
                                              data_prefix=data_prefix,
                                              min_freq=min_freq,
                                              max_vocab_size=max_vocab_size)
        self.min_len = min_len
        self.max_len = max_len
        self.share_vocab = share_vocab
        self.with_label = with_label

        self.SRC = TextField(tokenize_fn=tokenize,
                             embed_file=embed_file)
        # self.LABEL = NumberField(dtype = float)
        # self.LABEL = NumberField(sequential=True, dtype = int)

        if self.share_vocab:
            self.TGT = self.SRC
            self.CUE = self.SRC
        else:
            self.TGT = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)
            self.CUE = TextField(tokenize_fn=tokenize,
                                 embed_file=embed_file)

        if self.with_label:
            self.LABEL = NumberField(sequential=False, dtype = int)
            self.INDEX = NumberField(sequential=True, dtype = int)
            self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE, 'label': self.LABEL, 'index': self.INDEX}
        else:
            self.fields = {'src': self.SRC, 'tgt': self.TGT, 'cue': self.CUE }
            

        def src_filter_pred(src):
            """
            src_filter_pred
            """
            for sen in src:
                if not (min_len <= len(self.SRC.tokenize_fn(sen)) <= max_len):
                    return False
                else:
                    return True

        def tgt_filter_pred(tgt):
            """
            tgt_filter_pred
            """
            return min_len <= len(self.TGT.tokenize_fn(tgt)) <= max_len

        self.filter_pred = lambda ex: src_filter_pred(ex['src']) and tgt_filter_pred(ex['tgt'])
    def read_data(self, data_file, data_type="train"):
        """
        read_data
        """
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                # print(self.with_label)
                if self.with_label:#如果with_lable的话 这里的lable应该指选取persona条目的索引标签。 用于第二阶段。
                    query, response, personas, persona_label, key_index = line.strip().split('\t')[:5]
                    filter_personas = []
                    for sent in personas.split('**'):
                        filter_personas.append(' '.join(sent.split()[:self.max_len]))
                    index = key_index

                    data.append({'src': query, 'tgt': response, 'cue': filter_personas, 'label': persona_label, 'index': index})
                    '''
                    第二阶段
                    data=[{'src': "hi , how are you doing ? i am getting ready to do some cheetah chasing to stay in shape.",
                            'tgt': 'you must be very fast . hunting is one of my favorite hobbies .', 
                            'cue': ['i like to remodel homes', 'i like to go hunting', 'i like to shoot a bow'],
                            'label':'2', 
                            'index':'14 15 17'}
                            {},
                            {},
                            ...]
                    '''
                #没有persona_label的情况，用于第一阶段。
                else:
                    queries, response, persona = line.strip().split('\t')[:3]
                    src=queries.split('**')
                    # filter_persona = ' '.join(persona.split()[:self.max_len])
                    filter_persona = persona
                    data.append({'src': src, 'tgt': response, 'cue': filter_persona})
                    '''
                    第一阶段
                    data=[{'src': ["hi , how are you doing ? i am getting ready to do some <KEY> <KEY> to stay in <KEY> .",
                                   "i am doing well . getting ready for work , how about you ?",
                                   "getting <KEY> to play some <KEY> ! i am a <KEY> , got to stay <KEY> ."],
                            'tgt': 'you must be very fast . hunting is one of my favorite hobbies .', 
                            'cue': 'i like to go hunting',
                            'use': '1'}
                           {},
                           {},
                           ...]
                    '''

        filtered_num = len(data)
        if self.filter_pred is not None:
            data = [ex for ex in data if self.filter_pred(ex)]
        filtered_num -= len(data)
        print(
            "Read {} {} examples ({} filtered)".format(len(data), data_type.upper(), filtered_num))
        return data

    def read_data_multitask(self, data_file1, data_file2, data_type="train"):
        """
        read_data
        """
        data1 = []
        data2 = []
        with open(data_file2, "r", encoding="utf-8") as f:
            for line in f:
                # print(self.with_label)
                query, response, personas, persona_label, key_index = line.strip().split('\t')[:5]
                filter_personas = []
                for sent in personas.split('**'):
                    filter_personas.append(' '.join(sent.split()[:self.max_len]))
                index = key_index

                data2.append({'src': query, 'tgt': response, 'cue': filter_personas, 'label': persona_label, 'index': index})
        filtered_num = len(data2)
        if self.filter_pred is not None:
            data2 = [ex for ex in data2 if self.filter_pred(ex)]
        filtered_num -= len(data2)
        print(
            "Read {} {} examples ({} filtered)".format(len(data2), data_type.upper()+'task2', filtered_num))
        '''
        第二阶段
        data=[{'src': "hi , how are you doing ? i am getting ready to do some cheetah chasing to stay in shape.",
                'tgt': 'you must be very fast . hunting is one of my favorite hobbies .', 
                'cue': ['i like to remodel homes', 'i like to go hunting', 'i like to shoot a bow'],
                'label':'2', 
                'index':'14 15 17'}
                {},
                {},
                ...]
        '''
            #没有persona_label的情况，用于第一阶段。
        with open(data_file1, "r", encoding="utf-8") as f:
            for line in f:
                queries, response, persona = line.strip().split('\t')[:3]
                src=queries.split('**')
                # filter_persona = ' '.join(persona.split()[:self.max_len])
                filter_persona = persona
                data1.append({'src': src, 'tgt': response, 'cue': filter_persona})
        filtered_num = len(data1)
        if self.filter_pred is not None:
            data1 = [ex for ex in data1 if self.filter_pred(ex)]
        filtered_num -= len(data1)
        print(
            "Read {} {} examples ({} filtered)".format(len(data1), data_type.upper()+'task1', filtered_num))
        '''
        第一阶段
        data=[{'src': ["hi , how are you doing ? i am getting ready to do some <KEY> <KEY> to stay in <KEY> .",
                       "i am doing well . getting ready for work , how about you ?",
                       "getting <KEY> to play some <KEY> ! i am a <KEY> , got to stay <KEY> ."],
                'tgt': 'you must be very fast . hunting is one of my favorite hobbies .', 
                'cue': 'i like to go hunting',
                'use': '1'}
               {},
               {},
               ...]
        '''
        return data1, data2
