#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from modeling_graph_retriever import BertForGraphRetriever
from utils import DataProcessor
from utils import convert_examples_to_features
from utils import save, load
from utils import GraphRetrieverConfig
from utils import warmup_linear

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--task',
                        type=str,
                        default=None,
                        required = True,
                        help="Task code in {hotpot_open, hotpot_distractor, squad, nq}")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=378,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=5,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam. (def: 5e-5)")
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    # RNN graph retriever-specific parameters
    parser.add_argument("--example_limit",
                        default=None,
                        type=int)

    parser.add_argument("--max_para_num",
                        default=10,
                        type=int)
    parser.add_argument("--neg_chunk",
                        default=8,
                        type=int,
                        help="The chunk size of negative examples during training (to reduce GPU memory consumption with negative sampling)")
    parser.add_argument("--eval_chunk",
                        default=100000,
                        type=int,
                        help="The chunk size of evaluation examples (to reduce RAM consumption during evaluation)")
    parser.add_argument("--split_chunk",
                        default=300,
                        type=int,
                        help="The chunk size of BERT encoding during inference (to reduce GPU memory consumption)")

    parser.add_argument('--train_file_path',
                        type=str,
                        default=None,
                        help="File path to the training data")
    parser.add_argument('--dev_file_path',
                        type=str,
                        default=None,
                        help="File path to the eval data")

    parser.add_argument('--beam',
                        type=int,
                        default=1,
                        help="Beam size")
    parser.add_argument('--min_select_num',
                        type=int,
                        default=1,
                        help="Minimum number of selected paragraphs")
    parser.add_argument('--max_select_num',
                        type=int,
                        default=3,
                        help="Maximum number of selected paragraphs")
    parser.add_argument("--use_redundant",
                        action='store_true',
                        help="Whether to use simulated seqs (only for training)")
    parser.add_argument("--use_multiple_redundant",
                        action='store_true',
                        help="Whether to use multiple simulated seqs (only for training)")
    parser.add_argument('--max_redundant_num',
                        type=int,
                        default=100000,
                        help="Whether to limit the number of the initial TF-IDF pool (only for open-domain eval)")
    parser.add_argument("--no_links",
                        action='store_true',
                        help="Whether to omit any links (or in other words, only use TF-IDF-based paragraphs)")
    parser.add_argument("--pruning_by_links",
                        action='store_true',
                        help="Whether to do pruning by links (and top 1)")
    parser.add_argument("--expand_links",
                        action='store_true',
                        help="Whether to expand links with paragraphs in the same article (for NQ)")
    parser.add_argument('--tfidf_limit',
                        type=int,
                        default=None,
                        help="Whether to limit the number of the initial TF-IDF pool (only for open-domain eval)")

    parser.add_argument("--pred_file", default=None, type=str,
                        help="File name to write paragraph selection results")
    parser.add_argument("--tagme",
                        action='store_true',
                        help="Whether to use tagme at inference")
    parser.add_argument('--topk',
                        type=int,
                        default=2,
                        help="Whether to use how many paragraphs from the previous steps")
    
    parser.add_argument("--model_suffix", default=None, type=str,
                        help="Suffix to load a model file ('pytorch_model_' + suffix +'.bin')")

    parser.add_argument("--db_save_path", default=None, type=str,
                        help="File path to DB")
    
    args = parser.parse_args()

    cpu = torch.device('cpu')
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.train_file_path is not None:
        do_train = True

        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        os.makedirs(args.output_dir, exist_ok=True)

    elif args.dev_file_path is not None:
        do_train = False

    else:
        raise ValueError('One of train_file_path: {} or dev_file_path: {} must be non-None'.format(args.train_file_path, args.dev_file_path))
        
    processor = DataProcessor()

    # Configurations of the graph retriever
    graph_retriever_config = GraphRetrieverConfig(example_limit = args.example_limit,
                                                  task = args.task,
                                                  max_seq_length = args.max_seq_length,
                                                  max_select_num = args.max_select_num,
                                                  max_para_num = args.max_para_num,
                                                  tfidf_limit = args.tfidf_limit,

                                                  train_file_path = args.train_file_path,
                                                  use_redundant = args.use_redundant,
                                                  use_multiple_redundant = args.use_multiple_redundant,
                                                  max_redundant_num = args.max_redundant_num,

                                                  dev_file_path = args.dev_file_path,
                                                  beam = args.beam,
                                                  min_select_num = args.min_select_num,
                                                  no_links = args.no_links,
                                                  pruning_by_links = args.pruning_by_links,
                                                  expand_links = args.expand_links,
                                                  eval_chunk = args.eval_chunk,
                                                  tagme = args.tagme,
                                                  topk = args.topk,
                                                  db_save_path = args.db_save_path)

    logger.info(graph_retriever_config)
    
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)


    ##############################
    # Training                   #
    ##############################
    if do_train:
        model = BertForGraphRetriever.from_pretrained(args.bert_model,
                                                      cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1),
                                                      graph_retriever_config = graph_retriever_config)

        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

        POSITIVE = 1.0
        NEGATIVE = 0.0

        # Load training examples
        train_examples = None
        num_train_steps = None
        train_examples = processor.get_train_examples(graph_retriever_config)
        train_features = convert_examples_to_features(
            train_examples, args.max_seq_length, args.max_para_num, graph_retriever_config, tokenizer, train = True)

        # len(train_examples) and len(train_features) can be different, depedning on the redundant setting
        num_train_steps = int(
            len(train_features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total,
                             max_grad_norm = 1.0)
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        model.train()
        epc = 0
        for _ in range(int(args.num_train_epochs)):
            logger.info('Epoch '+str(epc+1))
            
            TOTAL_NUM = len(train_features)
            train_start_index = 0
            CHUNK_NUM = 10
            train_chunk = TOTAL_NUM//CHUNK_NUM
            chunk_index = 0

            random.shuffle(train_features)

            save_retry = False
            
            while train_start_index < TOTAL_NUM:                
                train_end_index = min(train_start_index+train_chunk-1, TOTAL_NUM-1)
                chunk_len = train_end_index - train_start_index + 1

                train_features_ = train_features[train_start_index:train_start_index+chunk_len]
                
                all_input_ids = torch.tensor([f.input_ids for f in train_features_], dtype=torch.long)
                all_input_masks = torch.tensor([f.input_masks for f in train_features_], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in train_features_], dtype=torch.long)
                all_output_masks = torch.tensor([f.output_masks for f in train_features_], dtype=torch.float)
                all_num_paragraphs = torch.tensor([f.num_paragraphs for f in train_features_], dtype=torch.long)
                all_num_steps = torch.tensor([f.num_steps for f in train_features_], dtype=torch.long)
                train_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_output_masks, all_num_paragraphs, all_num_steps)

                train_sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
                
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                logger.info('Examples from '+str(train_start_index)+' to '+str(train_end_index))
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    input_masks = batch[1]
                    batch_max_len = input_masks.sum(dim = 2).max().item()

                    num_paragraphs = batch[4]
                    batch_max_para_num = num_paragraphs.max().item()

                    num_steps = batch[5]
                    batch_max_steps = num_steps.max().item()

                    output_masks_cpu = (batch[3])[:, :batch_max_steps, :batch_max_para_num+1]

                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_masks, segment_ids, output_masks, _, __ = batch
                    B = input_ids.size(0)

                    input_ids = input_ids[:, :batch_max_para_num, :batch_max_len]
                    input_masks = input_masks[:, :batch_max_para_num, :batch_max_len]
                    segment_ids = segment_ids[:, :batch_max_para_num, :batch_max_len]
                    output_masks = output_masks[:, :batch_max_steps, :batch_max_para_num+1] # 1 for EOE

                    target = torch.FloatTensor(output_masks.size()).fill_(NEGATIVE) # (B, NUM_STEPS, |P|+1) <- 1 for EOE
                    for i in range(B):
                        output_masks[i, :num_steps[i], -1] = 1.0 # for EOE

                        for j in range(num_steps[i].item()-1):
                            target[i, j, j].fill_(POSITIVE)

                        target[i, num_steps[i]-1, -1].fill_(POSITIVE)
                    target = target.to(device)

                    neg_start = batch_max_steps-1
                    while neg_start < batch_max_para_num:
                        neg_end = min(neg_start+args.neg_chunk-1, batch_max_para_num-1)
                        neg_len = (neg_end-neg_start+1)

                        input_ids_ = torch.cat((input_ids[:, :batch_max_steps-1, :], input_ids[:, neg_start:neg_start+neg_len, :]), dim = 1)
                        input_masks_ = torch.cat((input_masks[:, :batch_max_steps-1, :], input_masks[:, neg_start:neg_start+neg_len, :]), dim = 1)
                        segment_ids_ = torch.cat((segment_ids[:, :batch_max_steps-1, :], segment_ids[:, neg_start:neg_start+neg_len, :]), dim = 1)
                        output_masks_ = torch.cat((output_masks[:, :, :batch_max_steps-1], output_masks[:, :, neg_start:neg_start+neg_len], output_masks[:, :, batch_max_para_num:batch_max_para_num+1]), dim = 2)
                        target_ = torch.cat((target[:, :, :batch_max_steps-1], target[:, :, neg_start:neg_start+neg_len], target[:, :, batch_max_para_num:batch_max_para_num+1]), dim = 2)

                        if neg_start != batch_max_steps-1:
                            output_masks_[:, :, :batch_max_steps-1] = 0.0
                            output_masks_[:, :, -1] = 0.0

                        loss = model(input_ids_, segment_ids_, input_masks_, output_masks_, target_, batch_max_steps)

                        if n_gpu > 1:
                            loss = loss.mean() # mean() to average on multi-gpu.
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        loss.backward()
                        tr_loss += loss.item()
                        neg_start = neg_end+1

                    nb_tr_examples += B
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # modify learning rate with special warm up BERT uses
                        lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                chunk_index += 1
                train_start_index = train_end_index + 1

                # Save the model at the half of the epoch
                if (chunk_index == CHUNK_NUM//2 or save_retry):
                    status = save(model, args.output_dir, str(epc+0.5))
                    save_retry = (not status)

                del all_input_ids
                del all_input_masks
                del all_segment_ids
                del all_output_masks
                del all_num_paragraphs
                del all_num_steps
                del train_data

            # Save the model at the end of the epoch
            save(model, args.output_dir, str(epc+1))

            epc += 1

    if do_train:
        return

    ##############################
    # Evaluation                 #
    ##############################
    assert args.model_suffix is not None

    if graph_retriever_config.db_save_path is not None:
        import sys
        sys.path.append('../')
        from pipeline.tfidf_retriever import TfidfRetriever
        tfidf_retriever = TfidfRetriever(graph_retriever_config.db_save_path, None)
    else:
        tfidf_retriever = None
        
    model_state_dict = load(args.output_dir, args.model_suffix)

    model = BertForGraphRetriever.from_pretrained(args.bert_model, state_dict=model_state_dict, graph_retriever_config = graph_retriever_config)
    model.to(device)

    model.eval()

    if args.pred_file is not None:
        pred_output = []
    
    eval_examples = processor.get_dev_examples(graph_retriever_config)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    TOTAL_NUM = len(eval_examples)
    eval_start_index = 0

    while eval_start_index < TOTAL_NUM:
        eval_end_index = min(eval_start_index+graph_retriever_config.eval_chunk-1, TOTAL_NUM-1)
        chunk_len = eval_end_index - eval_start_index + 1

        eval_features = convert_examples_to_features(
            eval_examples[eval_start_index:eval_start_index+chunk_len], args.max_seq_length, args.max_para_num, graph_retriever_config, tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_masks = torch.tensor([f.input_masks for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_output_masks = torch.tensor([f.output_masks for f in eval_features], dtype=torch.float)
        all_num_paragraphs = torch.tensor([f.num_paragraphs for f in eval_features], dtype=torch.long)
        all_num_steps = torch.tensor([f.num_steps for f in eval_features], dtype=torch.long)
        all_ex_indices = torch.tensor([f.ex_index for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_output_masks, all_num_paragraphs, all_num_steps, all_ex_indices)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        for input_ids, input_masks, segment_ids, output_masks, num_paragraphs, num_steps, ex_indices in tqdm(eval_dataloader, desc="Evaluating"):
            batch_max_len = input_masks.sum(dim = 2).max().item()
            batch_max_para_num = num_paragraphs.max().item()

            batch_max_steps = num_steps.max().item()

            input_ids = input_ids[:, :batch_max_para_num, :batch_max_len]
            input_masks = input_masks[:, :batch_max_para_num, :batch_max_len]
            segment_ids = segment_ids[:, :batch_max_para_num, :batch_max_len]
            output_masks = output_masks[:, :batch_max_para_num+2, :batch_max_para_num+1]
            output_masks[:, 1:, -1] = 1.0 # Ignore EOE in the first step

            input_ids = input_ids.to(device)
            input_masks = input_masks.to(device)
            segment_ids = segment_ids.to(device)
            output_masks = output_masks.to(device)

            examples = [eval_examples[eval_start_index+ex_indices[i].item()] for i in range(input_ids.size(0))]

            with torch.no_grad():
                pred, prob, topk_pred, topk_prob = model.beam_search(input_ids, segment_ids, input_masks, examples = examples, tokenizer = tokenizer, retriever = tfidf_retriever, split_chunk = args.split_chunk)
                        
            for i in range(len(pred)):
                e = examples[i]
                titles = [e.title_order[p] for p in pred[i]]

                # Output predictions to a file
                if args.pred_file is not None:
                    pred_output.append({})
                    pred_output[-1]['q_id'] = e.guid

                    pred_output[-1]['titles'] = titles
                    pred_output[-1]['probs'] = []
                    for prob_ in prob[i]:
                        entry = {'EOE': prob_[-1]}
                        for j in range(len(e.title_order)):
                            entry[e.title_order[j]] = prob_[j]
                        pred_output[-1]['probs'].append(entry)

                    topk_titles = [[e.title_order[p] for p in topk_pred[i][j]] for j in range(len(topk_pred[i]))]
                    pred_output[-1]['topk_titles'] = topk_titles

                    topk_probs = []
                    for k in range(len(topk_prob[i])):
                        topk_probs.append([])
                        for prob_ in topk_prob[i][k]:
                            entry = {'EOE': prob_[-1]}
                            for j in range(len(e.title_order)):
                                entry[e.title_order[j]] = prob_[j]
                            topk_probs[-1].append(entry)
                    pred_output[-1]['topk_probs'] = topk_probs

                    # Output the selected paragraphs
                    context = {}
                    for ts in topk_titles:
                        for t in ts:
                            context[t] = e.all_paras[t]
                    pred_output[-1]['context'] = context

        eval_start_index = eval_end_index + 1

        del eval_features
        del all_input_ids
        del all_input_masks
        del all_segment_ids
        del all_output_masks
        del all_num_paragraphs
        del all_num_steps
        del all_ex_indices
        del eval_data
        
    if args.pred_file is not None:
        json.dump(pred_output, open(args.pred_file, 'w'))
            
if __name__ == "__main__":
    main()
