from sequential_sentence_selector.modeling_sequential_sentence_selector import BertForSequentialSentenceSelector
from sequential_sentence_selector.run_sequential_sentence_selector import InputExample
from sequential_sentence_selector.run_sequential_sentence_selector import InputFeatures
from sequential_sentence_selector.run_sequential_sentence_selector import DataProcessor
from sequential_sentence_selector.run_sequential_sentence_selector import convert_examples_to_features

from pytorch_pretrained_bert.tokenization import BertTokenizer

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from tqdm import tqdm

class SequentialSentenceSelector:
    def __init__(self,
                 args,
                 device):

        if args.sequential_sentence_selector_path is None:
            return None
        
        print('initializing SequentialSentenceSelector...', flush=True)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_sequential_sentence_selector, do_lower_case=args.do_lower_case)    
        model_state_dict = torch.load(args.sequential_sentence_selector_path)
        self.model = BertForSequentialSentenceSelector.from_pretrained(args.bert_model_sequential_sentence_selector, state_dict=model_state_dict)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.processor = DataProcessor()
        print('Done!', flush=True)

    def convert_reader_output(self,
                              reader_output,
                              tfidf_retriever):
        new_output = []

        for data in reader_output:
            entry = {}
            entry['q_id'] = data['q_id']
            entry['question'] = data['question']
            entry['answer'] = data['answer']
            entry['titles'] = data['context']
            entry['context'] = tfidf_retriever.load_abstract_para_text(entry['titles'], keep_sentence_split = True)
            entry['supporting_facts'] = {t: [] for t in entry['titles']}
            new_output.append(entry)

        return new_output
        
    def predict(self,
                reader_output,
                tfidf_retriever,
                args):

        reader_output = self.convert_reader_output(reader_output, tfidf_retriever)
        eval_examples = self.processor.create_examples(reader_output)
        eval_features = convert_examples_to_features(
            eval_examples, args.max_seq_length_sequential_sentence_selector, args.max_sent_num, args.max_sf_num, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_masks = torch.tensor([f.input_masks for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_output_masks = torch.tensor([f.output_masks for f in eval_features], dtype=torch.float)
        all_num_sents = torch.tensor([f.num_sents for f in eval_features], dtype=torch.long)
        all_num_sfs = torch.tensor([f.num_sfs for f in eval_features], dtype=torch.long)
        all_ex_indices = torch.tensor([f.ex_index for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids,
                                  all_input_masks,
                                  all_segment_ids,
                                  all_output_masks,
                                  all_num_sents,
                                  all_num_sfs,
                                  all_ex_indices)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        pred_output = []

        for input_ids, input_masks, segment_ids, output_masks, num_sents, num_sfs, ex_indices in tqdm(eval_dataloader, desc="Evaluating"):
            batch_max_len = input_masks.sum(dim = 2).max().item()
            batch_max_sent_num = num_sents.max().item()
            batch_max_sf_num = num_sfs.max().item()

            input_ids = input_ids[:, :batch_max_sent_num, :batch_max_len]
            input_masks = input_masks[:, :batch_max_sent_num, :batch_max_len]
            segment_ids = segment_ids[:, :batch_max_sent_num, :batch_max_len]
            output_masks = output_masks[:, :batch_max_sent_num+2, :batch_max_sent_num+1]

            output_masks[:, 1:, -1] = 1.0 # Ignore EOE in the first step

            input_ids = input_ids.to(self.device)
            input_masks = input_masks.to(self.device)
            segment_ids = segment_ids.to(self.device)
            output_masks = output_masks.to(self.device)

            examples = [eval_examples[ex_indices[i].item()] for i in range(input_ids.size(0))]

            with torch.no_grad():
                pred, prob, topk_pred, topk_prob = self.model.beam_search(input_ids, segment_ids, input_masks, output_masks, max_num_steps = args.max_sf_num+1, examples = examples, beam = args.beam_sequential_sentence_selector)

            for i in range(len(pred)):
                e = examples[i]

                sfs = {}
                for p in pred[i]:
                    offset = 0
                    for j in range(len(e.titles)):
                        if p >= offset and p < offset+len(e.context[e.titles[j]]):
                            if e.titles[j] not in sfs:
                                sfs[e.titles[j]] = [[p-offset, e.context[e.titles[j]][p-offset]]]
                            else:
                                sfs[e.titles[j]].append([p-offset, e.context[e.titles[j]][p-offset]])
                            break
                        offset += len(e.context[e.titles[j]])

                # Hack
                for title in e.titles:
                    if title not in sfs and len(sfs) < 2:
                        sfs[title] = [0]

                output = {}
                output['q_id'] = e.guid
                output['supporting facts'] = sfs
                pred_output.append(output)

        return pred_output

