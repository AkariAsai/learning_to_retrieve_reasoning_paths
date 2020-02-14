import json
import os
import random

from tqdm import tqdm
import glob
import os

import torch

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class GraphRetrieverConfig:

    def __init__(self,
                 example_limit: int,
                 task: str,
                 max_seq_length: int,
                 max_select_num: int,
                 max_para_num: int,
                 tfidf_limit: int,

                 train_file_path: str,
                 use_redundant: bool,
                 use_multiple_redundant: bool,
                 max_redundant_num: int,

                 dev_file_path: str,
                 beam: int,
                 min_select_num: int,
                 no_links: bool,
                 pruning_by_links: bool,
                 expand_links: bool,
                 eval_chunk: int,
                 tagme: bool,
                 topk: int,
                 db_save_path: str):

        # General
        self.example_limit = example_limit

        self.open = False
        
        self.task = task
        assert task in ['hotpot_distractor', 'hotpot_open',
                        'squad', 'nq',
                        None]

        if task == 'hotpot_open' or (train_file_path is None and task in ['squad', 'nq']):
            self.open = True
        
        self.max_seq_length = max_seq_length
        
        self.max_select_num = max_select_num

        self.max_para_num = max_para_num

        self.tfidf_limit = tfidf_limit
        assert self.tfidf_limit is None or type(self.tfidf_limit) == int
        
        # Train
        self.train_file_path = train_file_path

        self.use_redundant = use_redundant

        self.use_multiple_redundant = use_multiple_redundant
        if self.use_multiple_redundant:
            self.use_redundant = True

        self.max_redundant_num = max_redundant_num
        assert self.max_redundant_num is None or self.max_redundant_num > 0 or not self.use_multiple_redundant
        
        # Eval
        self.dev_file_path = dev_file_path
        assert self.train_file_path is not None or self.dev_file_path is not None or task is None

        self.beam = beam

        self.min_select_num = min_select_num
        assert self.min_select_num >= 1 and self.min_select_num <= self.max_select_num

        self.no_links = no_links
        
        self.pruning_by_links = pruning_by_links
        if self.no_links:
            self.pruning_by_links = False

        self.expand_links = expand_links
        if self.no_links:
            self.expand_links = False
        
        self.eval_chunk = eval_chunk

        self.tagme = tagme

        self.topk = topk

        self.db_save_path = db_save_path

    def __str__(self):
        configStr = '\n\n' \
                    '### RNN graph retriever configurations ###\n' \
                    '@@ General\n' \
                    '- Example limit: ' + str(self.example_limit) + '\n' \
                    '- Task: ' + str(self.task) + '\n' \
                    '- Open: ' + str(self.open) + '\n' \
                    '- Max seq length: ' + str(self.max_seq_length) + '\n' \
                    '- Max select num: ' + str(self.max_select_num) + '\n' \
                    '- Max paragraph num (including links): ' + str(self.max_para_num) + '\n' \
                    '- Limit of the initial TF-IDF pool: ' + str(self.tfidf_limit) + '\n' \
                    '\n' \
                    '@@ Train\n' \
                    '- Train file path: ' + str(self.train_file_path) + '\n' \
                    '- Use redundant: ' + str(self.use_redundant) + '\n' \
                    '- Use multiple redundant: ' + str(self.use_multiple_redundant) + '\n' \
                    '- Max redundant num: ' + str(self.max_redundant_num) + '\n' \
                    '\n' \
                    '@@ Eval\n' \
                    '- Dev file path: ' + str(self.dev_file_path) + '\n' \
                    '- Beam size: ' + str(self.beam) + '\n' \
                    '- Min select num: ' + str(self.min_select_num) + '\n' \
                    '- No links: ' + str(self.no_links) + '\n' \
                    '- Pruning by links (and top 1): ' + str(self.pruning_by_links) + '\n' \
                    '- Exapnd links (for NQ): ' + str(self.expand_links) + '\n' \
                    '- Eval chunk: ' + str(self.eval_chunk) + '\n' \
                    '- Tagme: ' + str(self.tagme) + '\n' \
                    '- Top K: ' + str(self.topk) + '\n' \
                    '- DB save path: ' + str(self.db_save_path) + '\n' \
                    '#########################################\n'

        return configStr


class InputExample(object):

    def __init__(self, guid, q, c, para_dic, s_g, r_g, all_r_g, all_paras):

        self.guid = guid
        self.question = q
        self.context = c
        self.all_linked_paras_dic = para_dic
        self.short_gold = s_g
        self.redundant_gold = r_g
        self.all_redundant_gold = all_r_g
        self.all_paras = all_paras

        # paragraph index -> title
        self.title_order = []

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_masks, segment_ids, output_masks, num_paragraphs, num_steps, ex_index = None):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.output_masks = output_masks
        self.num_paragraphs = num_paragraphs
        self.num_steps = num_steps

        self.ex_index = ex_index

def expand_links(context, all_linked_paras_dic, all_paras):
    for context_title in context:
        # Paragraphs from the same article
        raw_context_title = context_title.split('_')[0]

        if context_title not in all_linked_paras_dic:
            all_linked_paras_dic[context_title] = {}

        for title in all_paras:
            if title == context_title or title in all_linked_paras_dic[context_title]:
                continue
            raw_title = title.split('_')[0]
            if raw_title == raw_context_title:
                all_linked_paras_dic[context_title][title] = all_paras[title]

        
class DataProcessor:

    def get_train_examples(self, graph_retriever_config):

        examples = []

        assert graph_retriever_config.train_file_path is not None

        file_name = graph_retriever_config.train_file_path

        if os.path.exists(file_name):
            examples += self._create_examples(file_name, graph_retriever_config, "train")
        else:
            file_list = list(glob.glob(file_name+'*'))
            for file_name in file_list:
                examples += self._create_examples(file_name, graph_retriever_config, "train")

        assert len(examples) > 0
        return examples
        
    def get_dev_examples(self, graph_retriever_config):

        examples = []

        assert graph_retriever_config.dev_file_path is not None

        file_name = graph_retriever_config.dev_file_path

        if os.path.exists(file_name):
            examples += self._create_examples(file_name, graph_retriever_config, "dev")
        else:
            file_list = list(glob.glob(file_name+'*'))
            for file_name in file_list:
                examples += self._create_examples(file_name, graph_retriever_config, "dev")

        assert len(examples) > 0
        return examples

    '''
    Read training examples from a json file
    * file_name: the json file name
    * graph_retriever_config: the graph retriever's configuration
    * task: a task name like "hotpot_open"
    * set_type: "train" or "dev"
    '''
    def _create_examples(self, file_name, graph_retriever_config, set_type):

        task = graph_retriever_config.task
        jsn = json.load(open(file_name, 'r'))
        
        examples = []

        '''
        Limit the number of examples used.
        This is mainly for sanity-chacking new settings.
        '''
        if graph_retriever_config.example_limit is not None:
            random.shuffle(jsn)
            jsn = sorted(jsn, key = lambda x: x['q_id'])
            jsn = jsn[:graph_retriever_config.example_limit]

        '''
        Find the mximum size of the initial context (links are not included)
        '''
        graph_retriever_config.max_context_size = 0
            
        logger.info('#### Loading examples... from {} ####'.format(file_name))
        for (_, data) in enumerate(tqdm(jsn, desc='Example')):

            guid = data['q_id']
            question = data['question']
            context = data['context'] # {context title: paragraph}
            all_linked_paras_dic = data['all_linked_paras_dic'] # {context title: {linked title: paragraph}}
            short_gold = data['short_gold'] # [title 1, title 2] (Both are gold)
            redundant_gold = data['redundant_gold'] # [title 1, title 2, title 3] ("title 1" is not gold)
            all_redundant_gold = data['all_redundant_gold']

            '''
            Limit the number of redundant examples
            '''
            all_redundant_gold = all_redundant_gold[:graph_retriever_config.max_redundant_num]

            '''
            Control the size of the initial TF-IDF retrieved paragraphs
            *** Training time: to take a blalance between TF-IDF-based and link-based negative examples ***
            '''
            if graph_retriever_config.tfidf_limit is not None:
                new_context = {}
                for title in context:
                    if len(new_context) == graph_retriever_config.tfidf_limit:
                        break
                    new_context[title] = context[title]
                context = new_context

            '''
            Use TagMe-based context at test time.
            '''
            if set_type == 'dev' and task == 'nq' and graph_retriever_config.tagme:
                assert 'tagged_context' in data

                '''
                Reformat "tagged_context" if needed (c.f. the "context" case above)
                '''
                if type(data['tagged_context']) == list:
                    tagged_context = {c[0]: c[1] for c in data['tagged_context']}
                    data['tagged_context'] = tagged_context

                '''
                Append valid paragraphs from "tagged_context" to "context"
                '''
                for tagged_title in data['tagged_context']:
                    tagged_text = data['tagged_context'][tagged_title]
                    if tagged_title not in context and tagged_title is not None and tagged_title.strip() != '' and tagged_text is not None and tagged_text.strip() != '':
                        context[tagged_title] = tagged_text

            '''
            Clean "context" by removing invalid paragraphs
            '''
            removed_keys = []
            for title in context:
                if title is None or title.strip() == '' or context[title] is None or context[title].strip() == '':
                    removed_keys.append(title)
            for key in removed_keys:
                context.pop(key)

            if task in ['squad', 'nq'] and set_type == 'train':
                new_context = {}

                orig_title = list(context.keys())[0].split('_')[0]
                
                orig_titles = []
                other_titles = []

                for title in context:
                    title_ = title.split('_')[0]

                    if title_ == orig_title:
                        orig_titles.append(title)
                    else:
                        other_titles.append(title)

                orig_index = 0
                other_index = 0

                while orig_index < len(orig_titles) or other_index < len(other_titles):
                    if orig_index < len(orig_titles):
                        new_context[orig_titles[orig_index]] = context[orig_titles[orig_index]]
                        orig_index += 1

                    if other_index < len(other_titles):
                        new_context[other_titles[other_index]] = context[other_titles[other_index]]
                        other_index += 1

                context = new_context


            '''
            Convert link format
            '''
            new_all_linked_paras_dic = {} # {context title: {linked title: paragraph}}

            all_linked_paras_dic # {linked_title: paragraph} or mixed
            all_linked_para_title_dic = data['all_linked_para_title_dic'] # {context_title: [linked_title_1, linked_title_2, ...]}

            removed_keys = []
            tmp = {}
            for key in all_linked_paras_dic:
                if type(all_linked_paras_dic[key]) == dict:
                    removed_keys.append(key)

                    for linked_title in all_linked_paras_dic[key]:
                        if linked_title not in all_linked_paras_dic:
                            tmp[linked_title] = all_linked_paras_dic[key][linked_title]

                        if key in all_linked_para_title_dic:
                            all_linked_para_title_dic[key].append(linked_title)
                        else:
                            all_linked_para_title_dic[key] = [linked_title]

            for key in removed_keys:
                all_linked_paras_dic.pop(key)

            for key in tmp:
                if key not in all_linked_paras_dic:
                    all_linked_paras_dic[key] = tmp[key]

            for context_title in context:
                if context_title not in all_linked_para_title_dic:
                    continue

                new_entry = {}

                for linked_title in all_linked_para_title_dic[context_title]:
                    if linked_title not in all_linked_paras_dic:
                        continue

                    new_entry[linked_title] = all_linked_paras_dic[linked_title]

                if len(new_entry) > 0:
                    new_all_linked_paras_dic[context_title] = new_entry

            all_linked_paras_dic = new_all_linked_paras_dic

            if set_type == 'dev':
                '''
                Clean "all_linked_paras_dic" by removing invalid paragraphs
                '''
                for c in all_linked_paras_dic:
                    removed_keys = []
                    links = all_linked_paras_dic[c]
                    for title in links:
                        if title is None or title.strip() == '' or links[title] is None or type(links[title]) != str or links[title].strip() == '':
                            removed_keys.append(title)
                    for key in removed_keys:
                        links.pop(key)

                all_paras = {}
                for title in context:
                    all_paras[title] = context[title]

                    if not graph_retriever_config.open:
                        continue
                    
                    if title not in all_linked_paras_dic:
                        continue
                    for title_ in all_linked_paras_dic[title]:
                        if title_ not in all_paras:
                            all_paras[title_] = all_linked_paras_dic[title][title_]
            else:
                all_paras = None

            if set_type == 'dev' and graph_retriever_config.expand_links:
                expand_links(context, all_linked_paras_dic, all_paras)

            if set_type == 'dev' and graph_retriever_config.no_links:
                all_linked_paras_dic = {}
                            
            graph_retriever_config.max_context_size = max(graph_retriever_config.max_context_size, len(context))

            '''
            Ensure that all the gold paragraphs are included in "context"
            '''
            if set_type == 'train':
                for t in short_gold + redundant_gold:
                    assert t in context
            
            examples.append(InputExample(guid = guid,
                                         q = question,
                                         c = context,
                                         para_dic = all_linked_paras_dic,
                                         s_g = short_gold,
                                         r_g = redundant_gold,
                                         all_r_g = all_redundant_gold,
                                         all_paras = all_paras))

        if set_type == 'dev':
            examples = sorted(examples, key = lambda x: len(x.all_paras))
        logger.info('Done!')
        
        return examples

def tokenize_question(question, tokenizer):
    tokens_q = tokenizer.tokenize(question)
    tokens_q = ['[CLS]'] + tokens_q + ['[SEP]']

    return tokens_q

def tokenize_paragraph(p, tokens_q, max_seq_length, tokenizer):
    tokens_p = tokenizer.tokenize(p)[:max_seq_length-len(tokens_q)-1]
    tokens_p = tokens_p + ['[SEP]']

    padding = [0] * (max_seq_length - len(tokens_p) - len(tokens_q))

    input_ids_ = tokenizer.convert_tokens_to_ids(tokens_q + tokens_p)
    input_masks_ = [1] * len(input_ids_)
    segment_ids_ = [0] * len(tokens_q) + [1] * len(tokens_p)

    input_ids_ += padding
    input_masks_ += padding
    segment_ids_ += padding

    assert len(input_ids_) == max_seq_length
    assert len(input_masks_) == max_seq_length
    assert len(segment_ids_) == max_seq_length

    return input_ids_, input_masks_, segment_ids_

def convert_examples_to_features(examples, max_seq_length, max_para_num, graph_retriever_config, tokenizer, train = False):
    """Loads a data file into a list of `InputBatch`s."""

    if not train and graph_retriever_config.db_save_path is not None:
        max_para_num = graph_retriever_config.max_context_size
        graph_retriever_config.max_para_num = max(graph_retriever_config.max_para_num, max_para_num)
    
    max_steps = graph_retriever_config.max_select_num
    
    DUMMY = [0] * max_seq_length
    features = []

    logger.info('#### Converting examples to features... ####')
    for (ex_index, example) in enumerate(tqdm(examples, desc='Example')):
        tokens_q = tokenize_question(example.question, tokenizer)
        
        ##############
        # Short gold #
        ##############
        title2index = {}
        input_ids = []
        input_masks = []
        segment_ids = []

        # Append gold and non-gold paragraphs from context
        if train and graph_retriever_config.use_redundant and len(example.redundant_gold) > 0:
            if graph_retriever_config.use_multiple_redundant:
                titles_list = example.short_gold + [redundant[0] for redundant in example.all_redundant_gold] + list(example.context.keys())
            else:
                titles_list = example.short_gold + [example.redundant_gold[0]] + list(example.context.keys())
        else:
            titles_list = example.short_gold + list(example.context.keys())
        for p in titles_list:

            if len(input_ids) == max_para_num:
                break

            # Avoid appending gold paragraphs as negative
            if p in title2index:
                continue

            # fullwiki eval
            # Gold paragraphs are not always in context
            if not train and graph_retriever_config.open and p not in example.context:
                continue
            
            title2index[p] = len(title2index)
            example.title_order.append(p)
            p = example.context[p]

            input_ids_, input_masks_, segment_ids_ = tokenize_paragraph(p, tokens_q, max_seq_length, tokenizer)
            input_ids.append(input_ids_)
            input_masks.append(input_masks_)
            segment_ids.append(segment_ids_)

        # Open-domain setting
        if graph_retriever_config.open:
            num_paragraphs_no_links = len(input_ids)
            
            for p_ in example.context:

                if not train and graph_retriever_config.db_save_path is not None:
                    break

                if len(input_ids) == max_para_num:
                    break

                if p_ not in example.all_linked_paras_dic:
                    continue
                
                for l in example.all_linked_paras_dic[p_]:

                    if len(input_ids) == max_para_num:
                        break
                    
                    if l in title2index:
                        continue

                    title2index[l] = len(title2index)
                    example.title_order.append(l)
                    p = example.all_linked_paras_dic[p_][l]

                    input_ids_, input_masks_, segment_ids_ = tokenize_paragraph(p, tokens_q, max_seq_length, tokenizer)
                    input_ids.append(input_ids_)
                    input_masks.append(input_masks_)
                    segment_ids.append(segment_ids_)
            
        assert len(input_ids) <= max_para_num
        
        num_paragraphs = len(input_ids)
        num_steps = len(example.short_gold)+1 # 1 for EOE

        if train:
            assert num_steps <= max_steps
        
        output_masks = [([1.0] * len(input_ids) + [0.0] * (max_para_num - len(input_ids) + 1)) for _ in range(max_para_num + 2)]

        if (not train) and graph_retriever_config.open:
            assert len(example.context) == num_paragraphs_no_links
            for i in range(len(output_masks[0])):
                if i >= num_paragraphs_no_links:
                    output_masks[0][i] = 0.0
        
        for i in range(len(input_ids)):
            output_masks[i+1][i] = 0.0            

        if train:
            size = num_steps-1

            for i in range(size):
                for j in range(size):
                    if i != j:
                        output_masks[i][j] = 0.0

            for i in range(size):
                output_masks[size][i] = 0.0
                        
            for i in range(max_steps):
                if i > size:
                    for j in range(len(output_masks[i])):
                        output_masks[i][j] = 0.0

            # Use REDUNDANT setting
            # Avoid treating the redundant paragraph as a negative example at the first step
            if graph_retriever_config.use_redundant and len(example.redundant_gold) > 0:
                if graph_retriever_config.use_multiple_redundant:
                    for redundant in example.all_redundant_gold:
                        output_masks[0][title2index[redundant[0]]] = 0.0
                else:
                    output_masks[0][title2index[example.redundant_gold[0]]] = 0.0
                    
        padding = [DUMMY] * (max_para_num - len(input_ids))
        input_ids += padding
        input_masks += padding
        segment_ids += padding

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_masks=input_masks,
                              segment_ids=segment_ids,
                              output_masks = output_masks,
                              num_paragraphs = num_paragraphs,
                              num_steps = num_steps,
                              ex_index = ex_index))

        if not train or not graph_retriever_config.use_redundant or len(example.redundant_gold) == 0:
            continue


        ##################
        # Redundant gold #
        ##################
        for redundant_gold in example.all_redundant_gold:
            hist = set()
            input_ids_r = []
            input_masks_r = []
            segment_ids_r = []

            # Append gold and non-gold paragraphs from context
            for p in redundant_gold + list(example.context.keys()):

                if len(input_ids_r) == max_para_num:
                    break

                #assert p in title2index
                if p not in title2index:
                    assert p not in redundant_gold
                    continue

                if p in hist:
                    continue
                hist.add(p)

                index = title2index[p]
                input_ids_r.append(input_ids[index])
                input_masks_r.append(input_masks[index])
                segment_ids_r.append(segment_ids[index])

            # Open-domain setting (mainly for HotpotQA fullwiki)
            if graph_retriever_config.open:

                for p in title2index:

                    if len(input_ids_r) == max_para_num:
                        break

                    if p in hist:
                        continue
                    hist.add(p)

                    index = title2index[p]
                    input_ids_r.append(input_ids[index])
                    input_masks_r.append(input_masks[index])
                    segment_ids_r.append(segment_ids[index])

            assert len(input_ids_r) <= max_para_num

            num_paragraphs_r = len(input_ids_r)
            num_steps_r = len(redundant_gold)+1

            assert num_steps_r <= max_steps

            output_masks_r = [([1.0] * len(input_ids_r) + [0.0] * (max_para_num - len(input_ids_r) + 1)) for _ in range(max_para_num + 2)]

            size = num_steps_r-1

            for i in range(size):
                for j in range(size):
                    if i != j:
                        output_masks_r[i][j] = 0.0

                if i > 0:
                    output_masks_r[i][0] = 1.0

            for i in range(size): #size-1
                output_masks_r[size][i] = 0.0

            for i in range(max_steps):
                if i > size:
                    for j in range(len(output_masks_r[i])):
                        output_masks_r[i][j] = 0.0

            padding = [DUMMY] * (max_para_num - len(input_ids_r))
            input_ids_r += padding
            input_masks_r += padding
            segment_ids_r += padding

            features.append(
                    InputFeatures(input_ids=input_ids_r,
                                  input_masks=input_masks_r,
                                  segment_ids=segment_ids_r,
                                  output_masks = output_masks_r,
                                  num_paragraphs = num_paragraphs_r,
                                  num_steps = num_steps_r,
                                  ex_index = None))

            if not graph_retriever_config.use_multiple_redundant:
                break

    logger.info('Done!')
    return features

def save(model, output_dir, suffix):
    logger.info('Saving the checkpoint...')
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(output_dir, "pytorch_model_"+suffix+".bin")

    status = True
    try:
        torch.save(model_to_save.state_dict(), output_model_file)
    except:
        status = False

    if status:
        logger.info('Successfully saved!')
    else:
        logger.warn('Failed!')
        
    return status

def load(output_dir, suffix):
    file_name = 'pytorch_model_' + suffix +'.bin'
    output_model_file = os.path.join(output_dir, file_name)
    return torch.load(output_model_file)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
