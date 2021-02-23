# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2020/3/21
@function: python interface for translate
"""
import os
import re
import logging
import json
import jieba
import tensorflow as tf
import sentencepiece as spm
from . import process
from html.parser import HTMLParser

logging.basicConfig(level=logging.INFO)
PATH=os.path.abspath('./models')

class Service:
    """python interface for translate"""
    def __init__(self,src_lang=None,tar_lang=None,version=None, use_gpu=True, batch_size=None):
        if src_lang is None:
            raise ValueError("Please specify Source language name, e.g. en, fr, de, zh ...")
        if tar_lang is None:
            raise ValueError("Please specify Target language name, e.g. en, fr, de, zh ...")
        if version is None:
            version = 1

        if use_gpu:
            self.device_list=self.get_device_list()
        else:
            self.device_list=["/cpu:0"]

        self.src=src_lang
        self.tar=tar_lang
        self.device_cnt=len(self.device_list)

        self.batch_size=batch_size
        if self.batch_size is None:
            self.batch_size=5120

        # specify model path
        model_path=os.path.join(PATH,"{}-{}/model/{}".format(src_lang,tar_lang,version))
        if not os.path.exists(model_path):
            raise ValueError("Invalid Path {}".format(model_path))

        # specify bpe path  add version
        source_bpe_path=os.path.join(PATH,"{}-{}/vocab/{}/{}.model".format(src_lang,tar_lang,version,src_lang))
        target_bpe_path=os.path.join(PATH,"{}-{}/vocab/{}/{}.model".format(src_lang,tar_lang,version,tar_lang))
        if not os.path.exists(source_bpe_path):
            raise FileNotFoundError("No found source vocab at {}".format(source_bpe_path))
        if not os.path.exists(target_bpe_path):
            raise FileNotFoundError("No found target vocab at {}".format(target_bpe_path))

        # specify vocab path
        source_vocab_path = os.path.join(PATH, "{}-{}/vocab/{}/{}.vocab".format(src_lang, tar_lang, version, src_lang))
        target_vocab_path = os.path.join(PATH, "{}-{}/vocab/{}/{}.vocab".format(src_lang, tar_lang, version, tar_lang))
        if not os.path.exists(source_vocab_path):
            raise FileNotFoundError("No found source vocab at {}".format(source_vocab_path))
        if not os.path.exists(target_vocab_path):
            raise FileNotFoundError("No found target vocab at {}".format(target_vocab_path))

        # model initailizer
        logging.info("Load model")
        self.sess,self.placeholders,self.output_placeholders=self.load_pb_model(model_path=model_path,
                                                                                device_list=self.device_list)

        # vocab initailizer
        logging.info("Load vocab")
        self.source_bpe_model=self.load_bpe_model(source_bpe_path)
        self.target_bpe_model=self.load_bpe_model(target_bpe_path)
        self.src_vocab = self.load_vocab(source_vocab_path, src=True)
        self.tar_vocab = self.load_vocab(target_vocab_path, src=False)

    def translate(self,sentences):
        """the batch of sentence"""
        cnt=len(sentences)

        # 1.preprocess text
        sentences=[self.preprocess(s,self.src) for s in sentences]

        # 2.tokenizer
        sentences=[self.tokenizer(s,self.source_bpe_model) for s in sentences]

        # 3.text2id
        token_ids=[self.sentence2tensor(sentence,self.src_vocab) for sentence in sentences]
        token_length=[len(tensor) for tensor in token_ids]

        # 4.sort
        sort_keys,sort_token_ids,sort_token_length=self.sort_inputs(token_ids,token_length)

        # 5.batch sample
        all_batch=self.batch_sampler(sort_token_ids,sort_token_length,batch_size=self.batch_size)
        # when the count of batchs cant be devided by the number of device list, will add  null character string
        if len(all_batch) % self.device_cnt != 0:
            padded_batch = ([[4]],[1])
            all_batch.extend([padded_batch] * (self.device_cnt - len(all_batch) % self.device_cnt))

        all_shard_items=[all_batch[i*self.device_cnt:(i+1)*self.device_cnt]
                          for i in range(len(all_batch)//self.device_cnt)]

        # 6.batch predict
        predictions=[]
        for shard_items in all_shard_items:
            shard_inputs=[item[0] for item in shard_items]
            shard_inputs_length=[item[1] for item in shard_items]

            # 7.batch predict
            shard_outputs=self.predictor(shard_inputs=shard_inputs,
                                         shard_inputs_length=shard_inputs_length,
                                         sess=self.sess,
                                         input_phds=self.placeholders,
                                         output_phds=self.output_placeholders)

            for outputs in shard_outputs:
                outputs=outputs[:,0,:].tolist()
                # 8.convert ids to text
                outputs=[self.tensor2sentence(t, tar_vocab=self.tar_vocab) for t in outputs]
                predictions.extend(outputs)

        # 9.postprocess
        sentences=[process.process_result(predictions[i]) for i in range(len(predictions))]
        if self.tar=="zh":
            sentences=[self.postprocess(sentence) for sentence in sentences]
        sentences=sentences[:cnt]

        # 10.post-sort
        sentences=[sentences[sort_keys[i]] for i in range(cnt)]
        return sentences

    @staticmethod
    def load_pb_model(model_path, device_list):
        inp_placeholders, inp_len_placeholders = [], []
        outp_placeholders = []
        graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with graph.as_default():
            sess=tf.Session(graph=graph, config=config)
            for i, device in enumerate(device_list):
                logging.info("Loading model on %s "%device[1:])
                with tf.variable_scope(tf.get_variable_scope(), reuse=(i != 0)):
                    with tf.device(device), tf.name_scope("parallel_%d" % i):
                        tf.saved_model.loader.load(sess=sess,
                                                   tags=[tf.saved_model.tag_constants.SERVING],
                                                   export_dir=model_path)
                        inputs = sess.graph.get_tensor_by_name('parallel_%d/source_0:0' % i)
                        inputs_length = sess.graph.get_tensor_by_name('parallel_%d/source_length_0:0' % i)
                        output = sess.graph.get_tensor_by_name('parallel_%d/parallel_0/strided_slice_6:0' % i)
                        inp_placeholders.append(inputs)
                        inp_len_placeholders.append(inputs_length)
                        outp_placeholders.append(output)
            placeholders = list(zip(inp_placeholders, inp_len_placeholders))
            return sess, placeholders, outp_placeholders

    @staticmethod
    def load_bpe_model(model_path):
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        return sp

    @staticmethod
    def load_vocab(vocab_path,src=True):
        word2index={}
        with open(vocab_path,encoding="utf8") as file:
            for line in file:
                word=line.strip().split()[0]
                if word not in word2index:
                    word2index[word]=len(word2index)
        index2word={v:k for k,v in word2index.items()}
        return word2index if src else index2word

    @staticmethod
    def predictor(shard_inputs, shard_inputs_length, sess, input_phds, output_phds):
        num_shard = len(output_phds)
        assert len(shard_inputs) == len(output_phds)
        feed_dict = {}
        for i in range(num_shard):
            # assgin op
            feed_dict[input_phds[i][0]] = shard_inputs[i]
            feed_dict[input_phds[i][1]] = shard_inputs_length[i]
        outputs = sess.run(output_phds, feed_dict=feed_dict)
        return outputs

    @staticmethod
    def tokenizer(sentence,bpe_model):
        tokens = bpe_model.EncodeAsPieces(sentence)
        sentence=" ".join(tokens)
        return sentence

    @staticmethod
    def sentence2tensor(sentence, src_vocab=None):
        piece = sentence.split()
        tensor=[]
        for p in piece:
            if p in src_vocab:
                tensor.append(src_vocab[p])
            else:
                tensor.append(src_vocab["<unk>"])
        return tensor+[src_vocab["<eos>"]]

    @staticmethod
    def tensor2sentence(tensor,tar_vocab=None):
        if not isinstance(tensor,list):
            tensor=tensor.tolist()
        sentence=[]
        for index in tensor:
            token=tar_vocab[index]
            if token=="<eos>": break
            if "▁" in token:
                sentence.append(token[1:])
            else:
                if len(sentence)==0:
                    sentence.append(token)
                else:
                    sentence[-1]+=token
        return " ".join(sentence)

    @staticmethod
    def padding(sentences,max_length):
        for sent in sentences:
            if len(sent) < max_length:
                sent.extend([0] * (max_length - len(sent)))
        return sentences

    @staticmethod
    def preprocess(sentence,src_lang=""):
        sentence=process.normalize(sentence,lang=src_lang)
        return sentence
    
    @staticmethod
    def postprocess(sentence):
        match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）] +(?<![a-zA-Z])|[0-9a-z A-Z]+')
        should_replace_list = match_regex.findall(sentence)
        order_replace_list = sorted(should_replace_list, key=lambda i: len(i), reverse=True)
        for i in order_replace_list:
            if i == u' ':
                continue
            new_i = i.strip()
            sentence = sentence.replace(i, new_i)
        return sentence

    @staticmethod
    def sort_inputs(token_ids, token_length):
        token_length=[(i,token_length[i]) for i in range(len(token_length))]
        sort_length=sorted(token_length,key=lambda x:x[1],reverse=False)
        sort_keys={}
        sort_token_length=[]
        for i,(origin_index,token_len) in enumerate(sort_length):
            sort_keys[origin_index]=i
            sort_token_length.append(token_len)
        sort_token_ids=[token_ids[index] for index in sort_keys]
        return sort_keys,sort_token_ids,sort_token_length

    @staticmethod
    def get_device_list():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
        if len(gpu_list) > 0:
            logging.info("GPU device count: %d"%len(gpu_list))
            return ["/gpu:%d" % d for d in range(len(gpu_list))]
        else:
            logging.info("Not found GPU device, will use CPU to compute")
            return ["/cpu:0"]

    def batch_sampler(self,sort_token_ids,sort_token_length,batch_size=6250):
        count=len(sort_token_length)
        all_batch=[]
        batch_ids,batch_length=[],[]
        for i in range(count):
            batch_ids.append(sort_token_ids[i])
            batch_length.append(sort_token_length[i])
            if len(batch_length)*batch_length[-1]>=batch_size or i==count-1 or len(batch_length)==768: # limit batch count
                # padding
                batch_ids=self.padding(batch_ids,max_length=max(batch_length))
                all_batch.append((batch_ids,batch_length))
                batch_ids, batch_length = [], []
        return all_batch
