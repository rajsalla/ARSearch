# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

"""
This script is modified from CodeBERT
https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/run.py
"""

from __future__ import absolute_import
import os
import sys
import math
import traceback
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from copy import deepcopy
from io import open
from itertools import cycle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import torch.nn as nn
from model import SimilarityClassifier



from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,WeightedRandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

from transformers import AutoTokenizer, AutoModel

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Pair(object):
    """A single text-code (nl-pl) pair."""
    def __init__(self,
                 idx,
                 simple_name,
                 thread_id,
                 text,
                 code
                 ):
        self.idx = idx
        self.simple_name = simple_name
        self.thread_id = thread_id
        self.text = text
        self.code = code

class LabeledPair(object):
    """A single text-code (nl-pl) pair."""
    def __init__(self,
                 idx,
                 simple_name,
                 thread_id,
                 text,
                 code,
                 api_comment,
                 api_implementation,
                 label
                 ):
        self.idx = idx
        self.simple_name = simple_name
        self.thread_id = thread_id
        self.text = text
        self.code = code
        self.cmt = api_comment
        self.impl = api_implementation
        self.label = label

def read_pairs(filename, mode="infer"):
    pairs = []
    if mode == "infer":
        with open(filename, "r") as fp:
            nl_pl_pairs = fp.readlines()
        for pair_info in nl_pl_pairs:
            pair_info = json.loads(pair_info)
            text, code = pair_info['pairs']
            pairs.append(
                Pair(
                    pair_info['idx'],
                    pair_info['simple_name'],
                    pair_info['thread_id'],
                    text,
                    code
                    )
            )
    elif mode == "train" or mode == "test" or mode == "eval":
        with open(filename, "r") as fp:
            training_pairs = fp.readlines()
        for training_info in training_pairs:
            training_info = json.loads(training_info)
            text, code = training_info['pairs']
            api_comment, api_implementation = training_info['doc_impl']
            label = training_info['cls_label']
            pairs.append(
                LabeledPair(
                    training_info['idx'],
                    training_info['simple_name'],
                    training_info['thread_id'],
                    text,
                    code,
                    api_comment,
                    api_implementation,
                    label
                )
            )
    return pairs

class PairFeatures(object):
    """
        A pair features combining text and code into source.
        For inference mode.
    """
    def __init__(self,
                 pair_id,
                 source_ids,
                 source_mask
                ):
        self.pair_id = pair_id
        self.source_ids = source_ids
        self.source_mask = source_mask

class LabeledPairFeatures(object):
    """
        A pair features combining text and code into source
        Has label. For training, testing and evaluating mode.
    """
    def __init__(self,
                 pair_id,
                 text_code_ids,
                 text_code_mask,
                 cmt_impl_ids,
                 cmt_impl_mask,
                 label
                ):
        self.pair_id = pair_id
        self.text_code_ids = text_code_ids
        self.text_code_mask = text_code_mask
        self.cmt_impl_ids = cmt_impl_ids
        self.cmt_impl_mask = cmt_impl_mask
        self.label = label

def truncate_pair(longer_one, shorter_one, args=None):
    total_length = len(longer_one) + len(shorter_one)
    delta = len(longer_one) - len(shorter_one)
    if len(shorter_one) < 200:
        # tokenized_test = tokenized_test[:-1*delta]
        longer_one = longer_one[:-1*(total_length-args.max_source_length+3)]
    else:
        longer_one = longer_one[:-1*delta]
        new_total_length = len(longer_one) + len(shorter_one)
        need_remove_tokens_length = new_total_length - args.max_source_length+3
        if need_remove_tokens_length % 2 == 0:
            longer_one = longer_one[:-1*int(need_remove_tokens_length/2)]
            shorter_one = shorter_one[:-1*int(need_remove_tokens_length/2)]
        else:
            longer_one = longer_one[:-1*int(need_remove_tokens_length/2+1)]
            shorter_one = shorter_one[:-1*int(need_remove_tokens_length/2)]
    return longer_one, shorter_one

def convert_pairs_to_features(pairs, tokenizer, args=None):
    features = []
    for pair_i, pair in enumerate(pairs):
        tokenized_text = tokenizer.tokenize(pair.text)
        tokenized_code = tokenizer.tokenize(pair.code)
        # truncate
        total_length = len(tokenized_text) + len(tokenized_code)
        if total_length > args.max_source_length -3:
            if len(tokenized_text) > len(tokenized_code):
                tokenized_text, tokenized_code = truncate_pair(tokenized_text, tokenized_code, args)
            else:
                tokenized_code, tokenized_text = truncate_pair(tokenized_code, tokenized_text, args)

        source_tokens = [tokenizer.cls_token]+tokenized_text+[tokenizer.sep_token]+tokenized_code+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
        features.append(
            PairFeatures(
                pair.idx,
                source_ids,
                source_mask
            )
        )
    return features

def get_training_features(pairs, tokenizer, args=None):
    def truncate(tokenized_text, tokenized_code):
        total_length = len(tokenized_text) + len(tokenized_code)
        if total_length > args.max_source_length -3:
            if len(tokenized_text) > len(tokenized_code):
                tokenized_text, tokenized_code = truncate_pair(tokenized_text, tokenized_code, args)
            else:
                tokenized_code, tokenized_text = truncate_pair(tokenized_code, tokenized_text, args)

        source_tokens = [tokenizer.cls_token]+tokenized_text+[tokenizer.sep_token]+tokenized_code+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
        return source_ids, source_mask

    features = []
    nrof_positives = 0
    nrof_negatives = 0

    for pair_i, pair in enumerate(pairs):
        tokenized_text = tokenizer.tokenize(pair.text)
        tokenized_code = tokenizer.tokenize(pair.code)
        tokenized_cmt = tokenizer.tokenize(pair.cmt)
        tokenized_impl = tokenizer.tokenize(pair.impl)
        # label = [0, 1] if pair.label else [1, 0] #binary classification if text_code is relevant to cmt_impl
        label = 1 if pair.label else 0 # For nn.CrossEntropyLoss label
        if label == 1:
            nrof_positives += 1
        else:
            nrof_negatives += 1

        # truncate
        text_code_ids, text_code_mask = truncate(tokenized_text, tokenized_code)
        cmt_impl_ids, cmt_impl_mask = truncate(tokenized_cmt, tokenized_impl)

        features.append(
            LabeledPairFeatures(
                pair.idx,
                text_code_ids,
                text_code_mask,
                cmt_impl_ids,
                cmt_impl_mask,
                label
            )
        )
    return features, nrof_positives, nrof_negatives

def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()
    


    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=512, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cuda", default="0", type=str,
                        help="GPU to run on")
    # print arguments
    args = parser.parse_args()
    logger.info(args)
    args.cuda = "cuda:"+args.cuda
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(args.cuda if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 1

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device

    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    
    # code2nl tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)

    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)
    model= SimilarityClassifier(encoder=encoder,config=config,
                                sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id,
                                cuda=args.cuda)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        torch.cuda.memory_summary()
        model.load_state_dict(torch.load(args.load_model_path,map_location=args.device))
        # model.load_state_dict(torch.load(args.load_model_path, map_location='cuda:1'))
    model.to(device)

        
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)


    def assign_token_indice(placeholder, list_subwords_indices):
        subword_indices_vector = deepcopy(placeholder)
        try:
            subword_indices_vector[:len(list_subwords_indices)] = list_subwords_indices
        except Exception as e:
            traceback.print_exc(e)
            exit()
        return subword_indices_vector

    activation = {}

    for child_i, child in enumerate(model.children()):
        for k,v in child.named_parameters():
            v.requires_grad = False

    root_path = "/app/facos/"
    infer_out_dir = os.path.join(root_path, "output/preprocess/")
    infer_file= os.path.join(root_path, "data/text_code_pairs_test.jsonl")
    file_path = os.path.join(infer_out_dir, "test_127_threads_result.json")
        
    os.makedirs(infer_out_dir, exist_ok=True)


    eval_examples = read_pairs(infer_file, mode="train")
    eval_features, _, _ = get_training_features(eval_examples, tokenizer, args)
    eval_pair_id = torch.tensor([f.pair_id for f in eval_features], dtype=torch.long)
    eval_text_code_ids = torch.tensor([f.text_code_ids for f in eval_features], dtype=torch.long)
    eval_text_code_mask = torch.tensor([f.text_code_mask for f in eval_features], dtype=torch.long)
    eval_cmt_impl_ids = torch.tensor([f.cmt_impl_ids for f in eval_features], dtype=torch.long)
    eval_cmt_impl_mask = torch.tensor([f.cmt_impl_mask for f in eval_features], dtype=torch.long)
    eval_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

    eval_dataset = TensorDataset(
                                eval_pair_id, 
                                eval_text_code_ids, 
                                eval_text_code_mask, 
                                eval_cmt_impl_ids, 
                                eval_cmt_impl_mask, 
                                eval_label)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)

    logger.info("\n***** Running inference *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.train_batch_size)

    model.eval()
    result_idx = 0
    output_results = {}
    softmax = nn.Softmax(dim=1)
    for eval_batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        eval_batch = tuple(t.to(device) for t in eval_batch)
        eval_pair_id, eval_text_code_ids, eval_text_code_mask, eval_cmt_impl_ids, eval_cmt_impl_mask, eval_label = eval_batch
        eval_preds = model(
                        eval_text_code_ids, 
                        eval_text_code_mask, 
                        eval_cmt_impl_ids, 
                        eval_cmt_impl_mask
                        )
        
        eval_preds = softmax(eval_preds)
        chosen_preds = torch.argmax(eval_preds, dim=1)
        for eval_i, each_label in enumerate(eval_label):
            eval_predict_result = chosen_preds[eval_i].cpu().detach().numpy()
            eval_preds_result = eval_preds[eval_i].cpu().detach().numpy()
            output_results[str(result_idx)] = [eval_predict_result.tolist(), eval_preds_result.tolist()]
            result_idx += 1
            
    # import codecs
    with open(file_path, "w+") as fp:
        json.dump(output_results, fp, indent=2)



if __name__ == "__main__":
    main()


