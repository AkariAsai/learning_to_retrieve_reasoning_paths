import json
from tqdm import tqdm

import torch

from graph_retriever.utils import InputExample
from graph_retriever.utils import InputFeatures
from graph_retriever.utils import tokenize_question
from graph_retriever.utils import tokenize_paragraph
from graph_retriever.utils import GraphRetrieverConfig
from graph_retriever.utils import expand_links
from graph_retriever.modeling_graph_retriever import BertForGraphRetriever

from pytorch_pretrained_bert.tokenization import BertTokenizer

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def create_examples(jsn, graph_retriever_config):

    task = graph_retriever_config.task

    examples = []

    '''
    Find the mximum size of the initial context (links are not included)
    '''
    graph_retriever_config.max_context_size = 0

    for data in jsn:

        guid = data['q_id']
        question = data['question']
        context = data['context'] # {context title: paragraph}
        all_linked_paras_dic = {} # {context title: {linked title: paragraph}}

        '''
        Use TagMe-based context at test time.
        '''
        if graph_retriever_config.tagme:
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

        all_paras = {}
        for title in context:
            all_paras[title] = context[title]

        if graph_retriever_config.expand_links:
            expand_links(context, all_linked_paras_dic, all_paras)
            
        graph_retriever_config.max_context_size = max(graph_retriever_config.max_context_size, len(context))

        examples.append(InputExample(guid = guid,
                                     q = question,
                                     c = context,
                                     para_dic = all_linked_paras_dic,
                                     s_g = None, r_g = None, all_r_g = None,
                                     all_paras = all_paras))

    return examples

def convert_examples_to_features(examples, max_seq_length, max_para_num, graph_retriever_config, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    max_para_num = graph_retriever_config.max_context_size
    graph_retriever_config.max_para_num = max(graph_retriever_config.max_para_num, max_para_num)
    
    max_steps = graph_retriever_config.max_select_num
    
    DUMMY = [0] * max_seq_length
    features = []

    for (ex_index, example) in enumerate(examples):
        tokens_q = tokenize_question(example.question, tokenizer)

        title2index = {}
        input_ids = []
        input_masks = []
        segment_ids = []

        titles_list = list(example.context.keys())
        for p in titles_list:

            if len(input_ids) == max_para_num:
                break

            if p in title2index:
                continue

            title2index[p] = len(title2index)
            example.title_order.append(p)
            p = example.context[p]

            input_ids_, input_masks_, segment_ids_ = tokenize_paragraph(p, tokens_q, max_seq_length, tokenizer)
            input_ids.append(input_ids_)
            input_masks.append(input_masks_)
            segment_ids.append(segment_ids_)

        num_paragraphs_no_links = len(input_ids)
            
        assert len(input_ids) <= max_para_num
        
        num_paragraphs = len(input_ids)
        
        output_masks = [([1.0] * len(input_ids) + [0.0] * (max_para_num - len(input_ids) + 1)) for _ in range(max_para_num + 2)]

        assert len(example.context) == num_paragraphs_no_links
        for i in range(len(output_masks[0])):
            if i >= num_paragraphs_no_links:
                output_masks[0][i] = 0.0
        
        for i in range(len(input_ids)):
            output_masks[i+1][i] = 0.0            
                    
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
                              num_steps = -1,
                              ex_index = ex_index))

    return features

class GraphRetriever:
    def __init__(self,
                 args,
                 device):

        self.graph_retriever_config = GraphRetrieverConfig(example_limit = None,
                                                           task = None,
                                                           max_seq_length = args.max_seq_length,
                                                           max_select_num = args.max_select_num,
                                                           max_para_num = args.max_para_num,
                                                           tfidf_limit = None,

                                                           train_file_path = None,
                                                           use_redundant = None,
                                                           use_multiple_redundant = None,
                                                           max_redundant_num = None,

                                                           dev_file_path = None,
                                                           beam = args.beam_graph_retriever,
                                                           min_select_num = args.min_select_num,
                                                           no_links = args.no_links,
                                                           pruning_by_links = args.pruning_by_links,
                                                           expand_links = args.expand_links,
                                                           eval_chunk = args.eval_chunk,
                                                           tagme = args.tagme,
                                                           topk = args.topk,
                                                           db_save_path = None)

        print('initializing GraphRetriever...', flush=True)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_graph_retriever, do_lower_case=args.do_lower_case)    
        model_state_dict = torch.load(args.graph_retriever_path)
        self.model = BertForGraphRetriever.from_pretrained(args.bert_model_graph_retriever, state_dict=model_state_dict, graph_retriever_config = self.graph_retriever_config)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        print('Done!', flush=True)

    def predict(self,
                tfidf_retrieval_output,
                retriever,
                args):

        pred_output = []
        
        eval_examples = create_examples(tfidf_retrieval_output, self.graph_retriever_config)

        TOTAL_NUM = len(eval_examples)
        eval_start_index = 0
        
        while eval_start_index < TOTAL_NUM:
            eval_end_index = min(eval_start_index+self.graph_retriever_config.eval_chunk-1, TOTAL_NUM-1)
            chunk_len = eval_end_index - eval_start_index + 1

            features = convert_examples_to_features(eval_examples[eval_start_index:eval_start_index+chunk_len], args.max_seq_length, args.max_para_num, self.graph_retriever_config, self.tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_masks = torch.tensor([f.input_masks for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_output_masks = torch.tensor([f.output_masks for f in features], dtype=torch.float)
            all_num_paragraphs = torch.tensor([f.num_paragraphs for f in features], dtype=torch.long)
            all_num_steps = torch.tensor([f.num_steps for f in features], dtype=torch.long)
            all_ex_indices = torch.tensor([f.ex_index for f in features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_output_masks, all_num_paragraphs, all_num_steps, all_ex_indices)

            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            logger.info('Examples from '+str(eval_start_index)+' to '+str(eval_end_index))
            for input_ids, input_masks, segment_ids, output_masks, num_paragraphs, num_steps, ex_indices in tqdm(eval_dataloader, desc="Evaluating"):

                batch_max_len = input_masks.sum(dim = 2).max().item()
                batch_max_para_num = num_paragraphs.max().item()
                batch_max_steps = num_steps.max().item()

                input_ids = input_ids[:, :batch_max_para_num, :batch_max_len]
                input_masks = input_masks[:, :batch_max_para_num, :batch_max_len]
                segment_ids = segment_ids[:, :batch_max_para_num, :batch_max_len]
                output_masks = output_masks[:, :batch_max_para_num+2, :batch_max_para_num+1]
                output_masks[:, 1:, -1] = 1.0 # Ignore EOS in the first step

                input_ids = input_ids.to(self.device)
                input_masks = input_masks.to(self.device)
                segment_ids = segment_ids.to(self.device)
                output_masks = output_masks.to(self.device)

                examples = [eval_examples[eval_start_index+ex_indices[i].item()] for i in range(input_ids.size(0))]

                with torch.no_grad():
                    pred, prob, topk_pred, topk_prob = self.model.beam_search(input_ids, segment_ids, input_masks, examples = examples, tokenizer = self.tokenizer, retriever = retriever, split_chunk = args.split_chunk)

                for i in range(len(pred)):
                    e = examples[i]

                    titles = [e.title_order[p] for p in pred[i]]
                    question = e.question

                    pred_output.append({})
                    pred_output[-1]['q_id'] = e.guid

                    pred_output[-1]['question'] = question

                    topk_titles = [[e.title_order[p] for p in topk_pred[i][j]] for j in range(len(topk_pred[i]))]
                    pred_output[-1]['topk_titles'] = topk_titles

                    topk_probs = []
                    pred_output[-1]['topk_probs'] = topk_probs

                    context = {}
                    context_from_tfidf = set()
                    context_from_hyperlink = set()
                    for ts in topk_titles:
                        for t in ts:
                            context[t] = e.all_paras[t]

                            if t in e.context:
                                context_from_tfidf.add(t)
                            else:
                                context_from_hyperlink.add(t)

                    pred_output[-1]['context'] = context
                    pred_output[-1]['context_from_tfidf'] = list(context_from_tfidf)
                    pred_output[-1]['context_from_hyperlink'] = list(context_from_hyperlink)
                    
            eval_start_index = eval_end_index + 1
            del features
            del all_input_ids
            del all_input_masks
            del all_segment_ids
            del all_output_masks
            del all_num_paragraphs
            del all_num_steps
            del all_ex_indices
            del eval_data
            
        return pred_output
