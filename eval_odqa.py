import argparse
import torch
import json
from tqdm import tqdm

from pipeline.tfidf_retriever import TfidfRetriever
from pipeline.graph_retriever import GraphRetriever
from pipeline.reader import Reader
from pipeline.sequential_sentence_selector import SequentialSentenceSelector

from eval_utils import read_jsonlines

# ODQA components for evaluation
class ODQAEval:
    def __init__(self):

        parser = argparse.ArgumentParser()

        ## Required parameters
        parser.add_argument("--eval_file_path",
                            default=None,
                            type=str,
                            required=True,
                            help="Eval data file path")
        parser.add_argument("--eval_file_path_sp",
                            default=None,
                            type=str,
                            required=False,
                            help="Eval data file path for supporting fact evaluation (only for HotpotQA)")
        parser.add_argument("--graph_retriever_path",
                            default=None,
                            type=str,
                            required=True,
                            help="Selector model path.")
        parser.add_argument("--reader_path",
                            default=None,
                            type=str,
                            required=True,
                            help="Reader model path.")
        parser.add_argument("--sequential_sentence_selector_path",
                            default=None,
                            type=str,
                            required=False,
                            help="supporting fact selector model path.")
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

        parser.add_argument("--bert_model_graph_retriever", default='bert-base-uncased', type=str,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                            "bert-base-multilingual-cased, bert-base-chinese.")

        parser.add_argument("--bert_model_sequential_sentence_selector", default='bert-base-uncased', type=str,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                            "bert-base-multilingual-cased, bert-base-chinese.")
        
        ## Other parameters
        parser.add_argument('--eval_batch_size',
                            type=int,
                            default=5,
                            help="Eval batch size")
        
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

        # RNN selector-specific parameters
        parser.add_argument("--max_para_num",
                            default=10,
                            type=int)

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
                            default=100,
                            help="Whether to limit the number of the initial TF-IDF pool (only for open-domain eval)")
        parser.add_argument('--pruning_l',
                            type=int,
                            default=10,
                            help="Set the maximum number of paragraphs retrieved from the same article.")

        parser.add_argument("--split_chunk", 
                            default=100, 
                            type=int,
                            help="Chunk size for BERT encoding at inference time")
    
        parser.add_argument("--eval_chunk", default=500, type=int,
                        help="Chunk size for inference of graph_retriever")
        
        # To use TagMe, you need to register first here https://sobigdata.d4science.org/web/tagme/tagme-help .
        parser.add_argument("--tagme",
                            action='store_true',
                            help="Whether to use tagme at inference")
        parser.add_argument("--tagme_api_key",
                            type=str,
                            default=None,
                            help="Set the TagMe private API key if you use TagMe.")
        
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
        parser.add_argument("--max_sent_num",
                            default=30,
                            type=int)
        parser.add_argument("--max_sf_num",
                            default=15,
                            type=int)
        # save intermediate results
        parser.add_argument('--tfidf_results_save_path',
                                type=str,
                                default=None,
                                help="If specified, the TF-IDF results will be saved in the file path")
        
        parser.add_argument('--selector_results_save_path',
                            type=str,
                            default=None,
                            help="If specified, the selector results will be saved in the file path")
        
        parser.add_argument('--reader_results_save_path',
                            type=str,
                            default=None,
                            help="If specified, the reader results will be saved in the file path")

        parser.add_argument('--sequence_sentence_selector_save_path',
                            type=str,
                            default=None,
                            help="If specified, the reader results will be saved in the file path")
        
        parser.add_argument('--saved_tfidf_retrieval_outputs_path',
                            type=str,
                            default=None,
                            help="If specified, load the saved TF-IDF retrieval results from the path.")

        parser.add_argument('--saved_selector_outputs_path',
                            type=str,
                            default=None,
                            help="If specified, load the saved reasoning path retrieval results from the path.")

        parser.add_argument("--sp_eval",
                            action='store_true',
                            help="set true if you evaluate supporting fact evaluations while running QA evaluation (HotpotQA only).")

        parser.add_argument("--sampled",
                            action='store_true',
                            help="evaluate on sampled examples; only for debugging and quick demo.")

        parser.add_argument("--use_full_article",
                            action='store_true',
                            help="Set true if you use all of the wikipedia paragraphs, not limiting to intro paragraphs.")

        parser.add_argument("--prune_after_agg",
                            action='store_true',
                            help="Pruning after aggregating all paragraphs from top k TFIDF paragraphs.")


        self.args = parser.parse_args()

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   and not self.args.no_cuda else "cpu")

        # Retriever
        self.retriever = TfidfRetriever(
            self.args.db_path, self.args.tfidf_path, self.args.use_full_article, self.args.pruning_l)

    def retrieve(self, eval_questions):
        tfidf_retrieval_output = []
        for _, eval_q in enumerate(tqdm(eval_questions, desc="Question")):
            if self.args.use_full_article is True:
                tfidf_retrieval_output += self.retriever.get_article_tfidf_with_hyperlinked_titles(
                    eval_q["id"], eval_q["question"], self.args)
            else:
                tfidf_retrieval_output += self.retriever.get_abstract_tfidf(
                    eval_q["id"], eval_q["question"], self.args)
        # create examples with retrieval results.
        print("retriever")
        print(len(tfidf_retrieval_output))
        
        # with `use_full_article` setting, we store the title to hyperlinked map and store 
        # it as retriever's property,
        if self.args.use_full_article is True:
            title2hyperlink_dic = {}
            for example in tfidf_retrieval_output:
                title2hyperlink_dic.update(
                    example["all_linked_para_title_dic"])
            self.retriever.store_title2hyperlink_dic(title2hyperlink_dic)
        
        return tfidf_retrieval_output
        
    def select(self, tfidf_retrieval_output):
        # Selector
        selector = GraphRetriever(self.args, self.device)

        selector_output = selector.predict(
            tfidf_retrieval_output, self.retriever, self.args)
        print("selector")
        print(len(selector_output))

        return selector_output
        
    def read(self, selector_output):
        # Reader
        reader = Reader(self.args, self.device)

        answers, titles  = reader.predict(selector_output, self.args)
        reader_output = {}
        print("reader")
        print(len(answers))
        print(answers)
        for s in selector_output:
            reader_output[s["q_id"]] = answers[s["q_id"]]

        return reader_output, titles
    
    def explain(self, reader_output_sp):
        sequential_sentence_selector = SequentialSentenceSelector(self.args, self.device)
        supporting_facts = sequential_sentence_selector.predict(
            reader_output_sp, self.retriever, self.args)

        return supporting_facts
        
        
    def eval(self):
        # load eval data 
        # the eval data is described in '{"id": "q_id", "question": "q1", answer": ["a11", ..., "a1i"]}' format (jsonlines) as in DrQA repository.
        # TODO: create eval data for HotpotQA, SQuAD and Natural Questions Open.
        eval_questions = read_jsonlines(self.args.eval_file_path)
        
        # Run (or load) graph retriever
        # FIXME: do not override saved results.
        tfidf_retrieval_output = None
        if self.args.saved_selector_outputs_path:
            selector_output = json.load(
                open(self.args.saved_selector_outputs_path))
        else:
            if self.args.saved_tfidf_retrieval_outputs_path:
                tfidf_retrieval_output = json.load(
                    open(self.args.saved_tfidf_retrieval_outputs_path))
            else:
                tfidf_retrieval_output = self.retrieve(eval_questions)
                
            selector_output = self.select(tfidf_retrieval_output)
        
        # read and extract answers from reasoning paths
        reader_output, titles = self.read(selector_output)
        
        if self.args.sequential_sentence_selector_path is None:
            return tfidf_retrieval_output, selector_output, reader_output
        else:
            reader_output_sp = [{'q_id': s['q_id'],
                            'question': s['question'],
                            'answer': reader_output[s['q_id']],
                            'context': titles[s['q_id']]} for s in selector_output]
            sp_selector_output = self.explain(reader_output_sp)
            print(sp_selector_output)
            return tfidf_retrieval_output, selector_output, reader_output, sp_selector_output


        
