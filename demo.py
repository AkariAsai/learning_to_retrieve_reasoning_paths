import argparse
import json

import torch

from pipeline.tfidf_retriever import TfidfRetriever
from pipeline.graph_retriever import GraphRetriever
from pipeline.reader import Reader
from pipeline.sequential_sentence_selector import SequentialSentenceSelector

import logging
class DisableLogger():
    def __enter__(self):
        logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
        logging.disable(logging.NOTSET)

class ODQA:
    def __init__(self, args):
        
        self.args = args
        
        device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")

        # TF-IDF Retriever
        self.tfidf_retriever = TfidfRetriever(self.args.db_path, self.args.tfidf_path)

        # Graph Retriever
        self.graph_retriever = GraphRetriever(self.args, device)

        # Reader
        self.reader = Reader(self.args, device)

        # Supporting facts selector
        self.sequential_sentence_selector = SequentialSentenceSelector(self.args, device)

    def predict(self,
                questions: list):

        print('-- Retrieving paragraphs by TF-IDF...', flush=True)
        tfidf_retrieval_output = []
        for i in range(len(questions)):
            question = questions[i]
            tfidf_retrieval_output += self.tfidf_retriever.get_abstract_tfidf('DEMO_{}'.format(i), question, self.args)

        print('-- Running the graph-based recurrent retriever model...', flush=True)
        graph_retrieval_output = self.graph_retriever.predict(tfidf_retrieval_output, self.tfidf_retriever, self.args)

        print('-- Running the reader model...', flush=True)
        answer, title = self.reader.predict(graph_retrieval_output, self.args)

        reader_output = [{'q_id': s['q_id'],
                          'question': s['question'],
                          'answer': answer[s['q_id']],
                          'context': title[s['q_id']]} for s in graph_retrieval_output]

        if self.args.sequential_sentence_selector_path is not None:
            print('-- Running the supporting facts retriever...', flush=True)
            supporting_facts = self.sequential_sentence_selector.predict(reader_output, self.tfidf_retriever, self.args)
        else:
            supporting_facts = []

        return tfidf_retrieval_output, graph_retrieval_output, reader_output, supporting_facts


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--graph_retriever_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Graph retriever model path.")
    parser.add_argument("--reader_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Reader model path.")
    parser.add_argument("--tfidf_path",
                        default=None,
                        type=str,
                        required=True,
                        help="TF-IDF path.")
    parser.add_argument("--db_path",
                        default=None,
                        type=str,
                        required=True,
                        help="DB path.")

    ## Other parameters
    parser.add_argument("--sequential_sentence_selector_path",
                        default=None,
                        type=str,
                        help="Supporting facts model path.")
    parser.add_argument("--max_sent_num",
                        default=30,
                        type=int)
    parser.add_argument("--max_sf_num",
                        default=15,
                        type=int)


    parser.add_argument("--bert_model_graph_retriever", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    
    parser.add_argument("--bert_model_sequential_sentence_selector", default='bert-large-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--max_seq_length",
                        default=378,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--max_seq_length_sequential_sentence_selector",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                        "Sequences longer than this will be truncated, and sequences shorter \n"
                        "than this will be padded.")
    
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    # RNN graph retriever-specific parameters
    parser.add_argument("--max_para_num",
                        default=10,
                        type=int)

    parser.add_argument('--eval_batch_size',
                        type=int,
                        default=5,
                        help="Eval batch size")

    parser.add_argument('--beam_graph_retriever',
                        type=int,
                        default=1,
                        help="Beam size for Graph Retriever")
    parser.add_argument('--beam_sequential_sentence_selector',
                        type=int,
                        default=1,
                        help="Beam size for Sequential Sentence Selector")
    
    parser.add_argument('--min_select_num',
                        type=int,
                        default=1,
                        help="Minimum number of selected paragraphs")
    parser.add_argument('--max_select_num',
                        type=int,
                        default=3,
                        help="Maximum number of selected paragraphs")
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

    parser.add_argument("--split_chunk", default=100, type=int,
                        help="Chunk size for BERT encoding at inference time")
    parser.add_argument("--eval_chunk", default=500, type=int,
                        help="Chunk size for inference of graph_retriever")
    
    parser.add_argument("--tagme",
                        action='store_true',
                        help="Whether to use tagme at inference")
    parser.add_argument('--topk',
                        type=int,
                        default=2,
                        help="Whether to use how many paragraphs from the previous steps")

    parser.add_argument("--n_best_size", default=5, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")


    odqa = ODQA(parser.parse_args())
    
    print()
    while True:
        questions = input('Questions: ')
        questions = questions.strip()
        if questions == 'q':
            break
        elif questions == '':
            continue

        questions = questions.strip().split('|||')
        tfidf_retrieval_output, graph_retriever_output, reader_output, supporting_facts = odqa.predict(questions)

        if graph_retriever_output is None:
            print()
            print('Invalid question! "{}"'.format(question))
            print()
            continue
        
        print()
        print('#### Retrieval results ####')
        print(json.dumps(graph_retriever_output, indent=4))
        print()
        
        print('#### Reader results ####')
        print(json.dumps(reader_output, indent=4))
        print()

        if len(supporting_facts) > 0:
            print('#### Supporting facts ####')
            print(json.dumps(supporting_facts, indent=4))
            print()
            

if __name__ == "__main__":
    with DisableLogger():
        main()
