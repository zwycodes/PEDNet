#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/field.py
"""

import re
import nltk
import torch
from tqdm import tqdm
from collections import Counter

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
NUM = "<num>"
KEY = "<key>"


def tokenize(s):
    """
    tokenize
    """
    # s = re.sub('\d+', NUM, s).lower() #用<num>标志替代数字字符[0-9] 
    s = s.lower()
    # tokens = nltk.RegexpTokenizer(r'\w+|<sil>|[^\w\s]+').tokenize(s)
    tokens = s.split(' ')#注意 split函数返回的是列表
    return tokens


class Field(object): #父类（python2.7，定义父类在括号内指定object）
    """
    Field
    """
    def __init__(self,
                 sequential=False,
                 dtype=None):#每当根据Field类创建新的实例时，__init__方法都会运行，这里定义了3个形参且形参self必不可少，因为调用__init__方法创建实例时，将自动传入实参self
        self.sequential = sequential
        self.dtype = dtype if dtype is not None else int #注意父类里数据类型提前定义了

    def str2num(self, string):
        """
        str2num
        """
        raise NotImplementedError

    def num2str(self, number):
        """
        num2str
        """
        raise NotImplementedError

    def numericalize(self, strings): #数字化
        """
        numericalize

        #strings有可能是一句话的字符串，也有可能是列表（列表元素是persona文本）
        """
        if isinstance(strings, str): #strings是一句话的字符串 也就是src/tgt/cue
            return self.str2num(strings) #isinstance与type不同点在type不考虑继承关系，而isinstance考虑
        else:#strings列表，也就是多句的query或者cue
            return [self.numericalize(s) for s in strings]

    def denumericalize(self, numbers): #在cuda中的数字化
        """                    
        denumericalize
        """
        if isinstance(numbers, torch.Tensor):
            with torch.cuda.device_of(numbers):
                numbers = numbers.tolist()#tolist()方法将数组或矩阵转换成列表
        if self.sequential:
            if not isinstance(numbers[0], list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]
        else:
            if not isinstance(numbers, list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]


class NumberField(Field): #子类
    """
    NumberField
    """
    def __init__(self,
                 sequential=False,
                 dtype=None):
        super(NumberField, self).__init__(sequential=sequential,
                                          dtype=dtype)    #（super函数需要两个实参，子类名和对象self，这是python2.7的语法；python3.6中super函数不需要写参数）

    def str2num(self, string):
        """
        str2num
        """
        if self.sequential:
            return [self.dtype(s) for s in string.split(" ")] #对label和key初始化的时候，dtype的值是float类型，也就是强制将一句话（字符串）的每一个word转换为float类型
        else:
            return self.dtype(string)

    def num2str(self, number):
        """
        num2str
        """
        if self.sequential: #str函数返回一个对象的string格式；join用于将序列中的元素以指定的字符连接生成一个新的字符串
            return " ".join([str(x) for x in number])  #如果对象是一个序列，将数字x转换成字符串并用空格连接生成新的字符串返回
        else:
            return str(number) #如果对象不是序列，直接将数字转换成字符串后返回


class TextField(Field):
    """
    TextField
    """
    def __init__(self,
                 tokenize_fn=None,
                 pad_token=PAD,
                 unk_token=UNK,
                 bos_token=BOS,
                 eos_token=EOS,
                 key_token=KEY,
                 special_tokens=None,
                 embed_file=None):
        super(TextField, self).__init__(sequential=True,
                                        dtype=int) #注意这里和子类NumberField的区别
        self.tokenize_fn = tokenize_fn if tokenize_fn is not None else str.split
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.key_token = key_token
        self.embed_file = embed_file

        specials = [self.pad_token, self.unk_token,
                    self.bos_token, self.eos_token, self.key_token]
        self.specials = [x for x in specials if x is not None]

        if special_tokens is not None:
            for token in special_tokens:#如果special_tokens中的某个特殊标志不在该类的specials列表中，将其加入到specials列表中
                if token not in self.specials:
                    self.specials.append(token)

        self.itos = []
        self.stoi = {}
        self.vocab_size = 0
        self.embeddings = None

    def build_vocab(self, texts, min_freq=0, max_size=None):#[src1,src2,src3...，tgt1,tgt2,tgt3..., cue1,cue2,cue3](原生未分词为列表-->字符串)
        """
        build_vocab

        texts = [src1,src2,src2,…, tgt1,tgt2,tgt3,…, cue1,cue1,cue3,…]
        """
        def flatten(xs):#这里xs是[src1,src2,src2,…, tgt1,tgt2,tgt3,…, cue1,cue1,cue3,…]
            """
            flatten 
            """
            flat_xs = []
            for x in xs:
                if isinstance(x, str):#如果x是字符串类型，则将x加入到flat_xs列表里面
                    flat_xs.append(x)
                elif isinstance(x[0], str):#否则如果x的第一个元素是字符串类型，则将x和flat_xs列表拼接
                    flat_xs += x#如果src1[0]是一个词，
                else:
                    flat_xs += flatten(x)
            return flat_xs

        # flatten texts
        texts = flatten(texts)
        '''
        将texts铺平成一个列表，这样texts中的元素全部为句子，texts = 
        ['hi , how are you doing ?', 'i am a 77 year old .', 'i love my family and animals .', ......]
        '''

        counter = Counter() #Counter类的目的是用来跟踪值出现的次数，这里创建一个空的Counter类
        for string in tqdm(texts):#原生未分词为列表-->字符串
            tokens = self.tokenize_fn(string)#[word1,word2,word3……](已分词-->列表)
            counter.update(tokens)#update是counter类中的方法，用来更新Counter，对于已有的元素计数加一，对没有的元素进行添加
        '''
        例如，打印counter：Counter({'i': 12, 'like': 8, 'to': 7, 'a': 5, '.': 4, 'you': 3, 'are': 2, 'am': 2, 
        'love': 2, 'my': 2, 'family': 2, 'remodel': 2, 'homes': 2, 'go': 2, 'hunting': 2, 'shoot': 2, 'bow': 2,
         'hi': 1, ',': 1, 'how': 1, 'doing': 1, '?': 1, '<num>': 1, 'year': 1, 'old': 1, 'and': 1, 'animals': 1, 
         'must': 1, 'be': 1, 'very': 1, 'fast': 1, 'it': 1, 'seems': 1, 'person': 1, 'home': 1, 'with': 1, 'the': 1, 
         'farms.': 1, 'compete': 1, 'at': 1, 'games': 1, 'parents': 1, 'have': 1, 'lot': 1, 'of': 1, 'pins': 1})
        '''

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in self.specials:#构建词表的时候特殊标记不计数
            del counter[tok]

        self.itos = list(self.specials)#list() 方法用于将元组转换为列表。注：元组与列表区别在于元组的元素值不能修改，元组是放在括号中，列表是放于方括号中。

        if max_size is not None:
            max_size = max_size + len(self.itos) #最大长度加特殊符号的长度

        # sort by frequency, then alphabetically
        # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作
        # list 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0]) #item函数返回（键，值）元组数组，按元组的第一个元素排序
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)#按词频大小降序排列，[('you', 2), ('?', 1), ('I', 1), ('love', 1), ('understand', 1)………………]

        cover = 0
        for word, freq in words_and_frequencies:
            if freq < min_freq :
                continue
            elif len(self.itos) == max_size:
                break
            self.itos.append(word) #itos=['<pad>', '<unk>', '<bos>', '<eos>','you','?','I',....] 除开特殊符号，其他按词频降序排列
            cover += freq  #达到最小频率的阈值的单词词频被加入到cover里面
        cover = cover / sum(freq for _, freq in words_and_frequencies)#计算限制词频后的单词频率总和占总单词频率总和的比例
        print(
            "Built vocabulary of size {} (coverage: {:.3f})".format(len(self.itos), cover)) #构建的词表对词频做了限制

        self.stoi = {tok: i for i, tok in enumerate(self.itos)} #词频限制后，是字典形式 {字符串：索引}
        self.vocab_size = len(self.itos) #词表长度（词频限制后）

        if self.embed_file is not None:
            self.embeddings = self.build_word_embeddings(self.embed_file) #见build_word_embeddings方法：[[* dim],[* dim],[* dim],[* dim]…………………………]

    def build_word_embeddings(self, embed_file):
        """
        build_word_embeddings
        """
        if isinstance(embed_file, list):
            embeds = [self.build_word_embeddings(e_file)
                      for e_file in embed_file]
        elif isinstance(embed_file, dict):
            embeds = {e_name: self.build_word_embeddings(e_file)
                      for e_name, e_file in embed_file.items()}
        else:
            cover = 0
            print("Building word embeddings from '{}' ...".format(embed_file))#embed_file文件的第一行是单词数目和特征维度
            with open(embed_file, "r", encoding="UTF-8") as f: #r表示只读
                # num, dim = map(int, f.readline().strip().split()) 读入num和嵌入维度；map根据提供的函数对指定序列做映射，python2返回列表，python3返回迭代器
                dim = 300
                embeds = [[0] * dim] * len(self.stoi)  #例如，2个单词5个特征维度，embeds=[[0,0,0,0,0],[0,0,0,0,0]]  #strip()移除字符串头尾指定的字符串，默认空格或换行符
                for line in f:
                    w, vs = line.rstrip().split(maxsplit=1)#rstrip() 删除每行多出来的空白行；分割成maxsplit+1个字符串（这里是2个）
                    if w in self.stoi: #w是词，vs是特征表示？（stoi是字典）
                        try:
                            vs = [float(x) for x in vs.split(" ")]
                        except Exception:
                            vs = []
                        if len(vs) == dim: #vs的列表长度等于嵌入维度的大小
                            embeds[self.stoi[w]] = vs #embeds[词汇索引]=特征表示
                            cover += 1
            rate = cover / len(embeds) #len(embeds)是词汇表的长度，cover是词汇表中包含的embed_file中的词汇个数（词汇表和embed_file词汇的交集）
            print("{} words have pretrained {}-D word embeddings (coverage: {:.3f})".format( \
                    cover, dim, rate))
        return embeds  #[[* dim],[* dim],[* dim],[* dim]…………………………] 元素按照stoi/itos也就是词频降序排列（len(self.stoi）embed_file中没有的单词embedding会设为0。

    def dump_vocab(self):
        """
        dump_vocab
        """
        vocab = {"itos": self.itos,
                 "embeddings": self.embeddings} #itos=['you','?','I',....]按词频降序排列
        return vocab

    def load_vocab(self, vocab):
        """
        load_vocab
        """
        self.itos = vocab["itos"] #self.itos=['you','?','I',....]按词频降序排列
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}  #self.stoi={'you':1,'?':2,'I':3,...}
        self.vocab_size = len(self.itos)
        self.embeddings = vocab["embeddings"]

    def str2num(self, string):#string是字符串，表示一句话，将这句话变成索引
        """
        str2num
        """
        #分别在句子首尾添加bos,eos字符。
        tokens = []
        unk_idx = self.stoi[self.unk_token] #unk_idx=1

        if self.bos_token:
            tokens.append(self.bos_token)#在句首添加bos字符

        tokens += self.tokenize_fn(string)#将string分词

        if self.eos_token:
            tokens.append(self.eos_token)#在句尾添加eos字符
        indices = [self.stoi.get(tok, unk_idx) for tok in tokens]#默认取1，也就是不在词表中的词，默认其索引为1
        return indices#索引化之后返回的是列表

    def num2str(self, number):
        """
        num2str
        """
        tokens = [self.itos[x] for x in number]
        if tokens[0] == self.bos_token:
            tokens = tokens[1:]
        text = []
        for w in tokens:
            if w != self.eos_token:
                text.append(w)
            else:
                break
        text = [w for w in text if w not in (self.pad_token, )]
        text = " ".join(text) #用空格连接，生成新的文本
        return text
