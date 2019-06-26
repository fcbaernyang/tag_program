# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import yaml
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.append("..")

from pytorch_pretrained_bert_1.tokenization import BertTokenizer
from pytorch_pretrained_bert_1.modeling import BertForTokenClassification,BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert import BertModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """
        用于装载数据集中的样本原始信息的容器类，包含的主要信息有句子(text_a)、谓词(text_b)和对应的标签(label)
    """


    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            text_a:字符串形式，原始句子，没有经过分词
            text_b:None
            label:字符串形式
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """
         用于装载进入模型前的样本信息的容器类，包含的主要的信息有input_ids,input_mask,segment_ids,label_id
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """
    从数据集文件读取数据集并将样本各个部分的原始信息加载待InputExample容器中的基类
    """

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


    @classmethod
    def _read_txt_tag(cls,input_file):
        """
        读取数据文件的内部方法
        根据是否是测试集数据有不同的操作
        """

        f=open(input_file,"r",encoding="utf-8")
        file=f.readlines()
        if file[0]=="train data\n" or file[0]=="dev data\n":
            #如果读取的是训练集或者是验证集需要同时读取文本和标签信息
            lines=[]
            tokens=[]
            tag=[]
            count=0
            for line in file[1:]:
                count=count+1
                if line=="\n":
                    l=["\t".join(tokens),tag]
                    lines.append(l)
                    tokens=[]
                    tag=[]
                else:
                    tokens.append(line.split("\t")[0].strip())
                    tag.append(line.split("\t")[1].strip())
            return lines
        elif file[0]=="test data\n":
            #如果读取的是测试集，需要读取文本信息
            lines=[]
            tokens=[]
            tag=[]
            for line in file[1:]:
                
                if line=="\n":
                    l=["\t".join(tokens),tag]
                    lines.append(l)
                    tokens=[]
                    tag=[]
                else:
                    tokens.append(line.split("\t")[0].strip())
                    tag.append("O")
            """
                line_1=line.strip()
                tokens=("\t".join(open_dataset_char(line_1)))
                for i in range(len(open_dataset_char(line_1))):
                    tag.append("O")
                lines.append([tokens,tag])
            """
            return lines




class TagProcessor(DataProcessor):
    #用于处理序列标注任务的子类
    def get_train_examples(self,data_dir):
        return self._create_examples(
                self._read_txt_tag(data_dir),"train"),self._read_txt_tag(data_dir)

    def get_dev_examples(self,data_dir):
        return self._create_examples(
                self._read_txt_tag(data_dir),"dev"),self._read_txt_tag(data_dir)

    def get_test_examples(self,data_dir):
        return self._create_examples(
                self._read_txt_tag(data_dir),"test"),self._read_txt_tag(data_dir)

    """
    def get_labels(self):
        self._read_txt_tag(data_dir)
        return ["B-mm_option","I-mm_option","O"]
    """

    def _create_examples(self,lines,set_type):
        examples=[]
        for line in lines:
            guid="%s-%s"
            text_a =line[0]

            label=line[1]
            examples.append(
                    InputExample(guid=guid,text_a=text_a,text_b=None,label=label))

        return examples


def get_preds(logits):
    #将模型输出的预测信息转换为具体的预测标签值的函数
    preds=[]
    for one_batch_logits in logits:
        for line in one_batch_logits:
            preds.append(torch.nn.functional.softmax(line).detach().cpu().numpy().tolist())

    preds_1 = [np.argsort(pred)[::-1][0] for pred in preds]
    preds_1= np.array(preds_1)
    preds_1=torch.from_numpy(preds_1)
     
    return preds_1

def mult_class_metrics(preds,label_ids):
    #对预测结果进行指标判断的方法
    """
    result=[]
    preds=np.array(preds)
    top1= preds
    test_y_numerical=top1
    for label_list in label_ids:
        result.append(label_list)

        #for token_label in label_list:
           # result.append(token_label)
    a = accuracy_score(test_y_numerical, result)
    

    b = f1_score(test_y_numerical, result, average="macro")
    c = f1_score(test_y_numerical, result, average="micro")
    d = precision_score(test_y_numerical, result, average="macro")
    e = precision_score(test_y_numerical, result, average="micro")
    f = recall_score(test_y_numerical, result, average="macro")
    g = recall_score(test_y_numerical, result, average="micro")
   """
    a = f1_score(preds,label_ids,average="macro")
    return a

def open_dataset_char(sentence):
    #用于对语句进行分字的函数，对英文字符和数字进行了处理，将英文单词和数字作为整体进行分割

    sentence=sentence.strip()
    char_list=[]
    ss=""
    for s in sentence:
        if s.islower() or  s.isupper()  or s.isdigit():
            ss=ss+s
        elif s==' ':
            if len(ss)!=0:
                char_list.append(ss)
                ss=""
        else:
            if len(ss)!=0:
                char_list.append(ss)
                ss=""
            char_list.append(s)
    if len(ss)!=0:
        char_list.append(ss)

    return char_list

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """
    将InputExample中的信息转换成InputFeature
    """

    label_map = {label : i  for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        #tokens_a = open_dataset_char(example.text_a)
        tokens_a=example.text_a.split("\t")

        #print(len(example.label))
        #print(example.text_a.split("\t")
        #print(example.text_a)
        #print(tokens_a)
        tokens_b = None
    
        if example.text_b:
            tokens_b = open_dataset_char(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
        #label_id = [label_map[i] for i in example.label]
        #label_id = [label_map["O"]]+label_id+[label_map["O"]]
        #segment_ids = [0] * len(tokens)
        label_list=["O"]+example.label+["O"]
        tokens_a_input_ids,label_id=tokenizer.convert_tokens_to_ids(tokens_a,label_list,label_map)
        segment_ids =[0]*len(tokens_a_input_ids)
        """
        if tokens_b:
            tokens_b = tokens_b + ["[SEP]"]
            tokens_b_input_ids=tokenizer.convert_tokens_to_ids(tokens_b,example.label,label_map)
            segment_ids += [1] * (len(tokens_b_input_ids) + 1)
        """
        #if tokens_b:

           # input_ids = tokens_a_input_ids+tokens_b_input_ids
        #else:
        input_ids = tokens_a_input_ids

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id +=padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_id) == max_seq_length
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features,label_map


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
"""
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)
"""
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--config_dir",
                        default=None,
                        type=str,
                        required=True,
                        )


    ## Other parameters
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()
    config_dir=args.config_dir
    f=open(config_dir)
    config=yaml.load(f)
    #将配置文件中的信息形成相应的变量
    do_train=config["do_train"]
    do_eval=config["do_eval"]
    do_test=config["do_test"]
    train_data_dir=config["train_data_dir"]
    dev_data_dir = config["dev_data_dir"]
    test_data_dir = config["test_data_dir"]
    label_list_dir = config["label_list_dir"]
    bert_config_dir = config["bert_config_dir"]
    tokenizer_dir = config["tokenizer_dir"]
    origin_bert_model_dir = config["origin_bert_model_dir"]
    bert_model_output_dir = config["bert_model_output_dir"]
    predict_file_dir = config["predict_file_dir"]
    max_seq_length =int(config["max_seq_length"])
    train_batch_size =int(config["train_batch_size"])
    eval_batch_size = int(config["eval_batch_size"])
    test_batch_size = int(config["test_batch_size"])
    learning_rate = float(config["learning_rate"])
    num_train_epoch = int(config["num_train_epoch"])
    cuda_use = config["cuda_use"]
    do_lower_case = config["do_lower_case"]


    #根据对应的配置信息决定是否使用GPU
    if args.local_rank == -1 or cuda_use:
        device = torch.device("cuda" if torch.cuda.is_available() and cuda_use else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    train_batch_size = int(train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    #if not args.do_train and not args.do_eval:
     #   raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
       # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
   # os.makedirs(args.output_dir, exist_ok=True)

    #实例化处理数据集的类
    processor = TagProcessor()
    """
    num_labels = 3
    label_list = processor.get_labels()
    """
    #实例化分字的类
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir, do_lower_case=do_lower_case)

    train_examples = None
    num_train_steps = None
    if do_train:
        #得到训练集的InputExample
        train_examples ,train_process= processor.get_train_examples(train_data_dir)
        _,dev_process = processor.get_dev_examples(dev_data_dir)
        train_tag=[]
        #根据训练集和验证集的标签得到label list,将label list写入文件
        for line in train_process:
            train_tag.extend(line[1])

        for line_1 in dev_process:
            train_tag.extend(line_1[1])
        #print(train_tag)
        label_list=np.unique(train_tag).tolist()
        num_labels=len(label_list)
        f=open(label_list_dir,'w',encoding="utf-8")
        for i in label_list:
            f.write(i+'\n')
        f.close()
        #得到训练步长
        num_train_steps = int(
            len(train_examples) / train_batch_size / args.gradient_accumulation_steps * num_train_epoch)


    if do_eval:
        #得到验证集的InputExample
        eval_examples,_ = processor.get_dev_examples(dev_data_dir)
        #num_train_steps = int(
           # len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    if do_test:
        #读取label list的文件
        f=open(label_list_dir)
        label_list_n=f.readlines()
        label_list=[label.strip() for label in label_list_n]
        num_labels=len(label_list)
        #得到测试集的InputExample和测试集的句子列表
        test_examples,test_process = processor.get_test_examples(test_data_dir)
        test_sentence=[line[0] for line in test_process]
    bert_config = BertConfig.from_json_file(bert_config_dir)
    # Prepare model
    #加载预训练模型
    model = BertForTokenClassification.from_pretrained(origin_bert_model_dir,num_labels = num_labels)


    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    #准备优化器
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0

    if do_train:
        #得到训练集的InputFeature
        train_features,label_map = convert_examples_to_features(
            train_examples, label_list, max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        #将InputFeature中的各个部分信息装入Tensor中
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)


        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
        model.train()
        best_acc=0
        #进入训练轮次的循环
        for epoch in trange(int(num_train_epoch), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            batch_count=0
            #进入一轮训练的batch循环
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch_count=batch_count+1
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                #在一个batch中得到loss
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                #根据loss进行优化
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            #add  dev
                #if batch_count %5000 ==0:

            if do_eval==True:
                eval_features ,label_map= convert_examples_to_features(
                eval_examples, label_list, max_seq_length, tokenizer)
                all_input_ids_eval = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask_eval = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids_eval = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids_eval = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids_eval, all_input_mask_eval, all_segment_ids_eval, all_label_ids_eval)

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler , batch_size=eval_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                preds=[]
                dev_label_id=[]
                count=0
                count_1=0
                for input_ids_eval, input_mask_eval, segment_ids_eval, label_ids_eval in eval_dataloader:
                    input_ids_eval = input_ids_eval.to(device)
                    input_mask_eval = input_mask_eval.to(device)
                    segment_ids_eval = segment_ids_eval.to(device)
                    label_ids_eval = label_ids_eval.to(device)
                    with torch.no_grad():
                        logits = model(input_ids_eval, segment_ids_eval, input_mask_eval)
                    active_loss = input_mask_eval.view(-1) == 1
                    preds_batch=get_preds(logits)
                    active_preds_batch=preds_batch.view(-1)[active_loss]
                    active_label_ids_eval=label_ids_eval.view(-1)[active_loss]
                    preds.extend(active_preds_batch.to("cpu").numpy().tolist())
                    dev_label_id.extend(active_label_ids_eval.to('cpu').numpy().tolist())
                #对验证集的预测信息进行指标判定
                metric= mult_class_metrics(preds,dev_label_id)
                #在每一轮训练完成得到的模型如果在验证集上的指标比历史最好的指标好，就保存模型
 
                if metric>best_acc:
                    best_acc=metric
                    print("best_acc is:"+str(metric)+" save the model")
                # Save a trained model
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = bert_model_output_dir
                    torch.save(model_to_save.state_dict(), output_model_file)
        if do_eval==False:
            #如果不进行验证，模型训练后直接保存模型
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = bert_model_output_dir
            torch.save(model_to_save.state_dict(), output_model_file)
    



    if do_test:
        f=open(label_list_dir)
        label_list_n=f.readlines()
        label_list=[label.strip() for label in label_list_n]
        num_labels=len(label_list)

        #加载经过训练得到的模型
        model_state_dict=torch.load(bert_model_output_dir)
        model=BertForTokenClassification.from_pretrained(origin_bert_model_dir,state_dict=model_state_dict,num_labels=num_labels)
        model.to(device)
        #得到测试集的InputFeature
        test_features,label_map = convert_examples_to_features(
        test_examples, label_list, max_seq_length, tokenizer)
        all_input_ids_test = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask_test = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids_test = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids_test = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids_test, all_input_mask_test, all_segment_ids_test, all_label_ids_test)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler,batch_size=test_batch_size)

        model.eval()
        preds=[]
        count=0
        count_1=0
        for input_ids_test, input_mask_test, segment_ids_test, label_ids_test in test_dataloader:
            input_ids_test = input_ids_test.to(device)
            input_mask_test = input_mask_test.to(device)
            segment_ids_test = segment_ids_test.to(device)
            #label_ids_test = label_ids_test.to(device)
            with torch.no_grad():
                logits = model(input_ids_test, segment_ids_test, input_mask_test)
            active_loss = input_mask_test.view(-1) == 1
            preds_batch=get_preds(logits)
            active_preds_batch=preds_batch.view(-1)[active_loss]
            #active_label_ids_test=label_ids_test.view(-1)[active_loss]
            preds.extend(active_preds_batch.to("cpu").numpy().tolist())
            #test_label_id.extend(active_label_ids_test.to('cpu').numpy().tolist())
            label_dict={}
            for key,value in label_map.items():
                label_dict[value]=key
        preds_label=[label_dict.get(i) for i in preds]
        c=0
        for s in test_sentence:
            for char in s.split("\t"):
                c=c+1
        #将预测结果写入文件
        f=open(predict_file_dir,'w',encoding="utf-8")
        f.write("test predict file"+'\n')
        f.write("\n")
        count=0
        for sentence in test_sentence:
            count=count+1
            for char in sentence.split("\t"):
                f.write(char+"\t"+preds_label[count]+"\n")
                count=count+1
            count=count+1
            f.write("\n")
    
        f.close()





if __name__ == "__main__":
    main()
