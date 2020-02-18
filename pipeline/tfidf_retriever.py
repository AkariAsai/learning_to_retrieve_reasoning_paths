import json

from retriever.doc_db import DocDB
from retriever.tfidf_doc_ranker import TfidfDocRanker
from retriever.tfidf_vectorizer_article import TopTfIdf

from retriever.utils import load_para_collections_from_tfidf_id_intro_only, \
    load_para_and_linked_titles_dict_from_tfidf_id, prune_top_k_paragraphs, \
    normalize
    
from retriever.tfidf_vectorizer_article import TopTfIdf

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class TfidfRetriever:
    def __init__(self,
                 db_save_path: str,
                 tfidf_model_path: str,
                 use_full_article: bool = False,
                 pruning_l: int = 10):

        print('initializing TfidfRetriever...', flush=True)
        self.db = DocDB(db_save_path)
        if tfidf_model_path is not None:
            self.ranker = TfidfDocRanker(tfidf_path=tfidf_model_path, strict=False)
        self.use_full_article = use_full_article
        # FIXME: make this to dynamically set the pruning_l
        self.pruning_l = pruning_l
        print('Done!', flush=True)

    def store_title2hyperlink_dic(self, title2hyperlink_dic):
        self.title2hyperlink_dic = title2hyperlink_dic
        
    def load_abstract_para_text(self,
                                doc_names,
                                keep_sentence_split = False):
        
        context = {}

        for doc_name in doc_names:
            para_title_text_pairs = load_para_collections_from_tfidf_id_intro_only(doc_name, self.db)
            if len(para_title_text_pairs) == 0:
                logger.warning("{} is missing".format(doc_name))
                continue
            else:
                para_title_text_pairs = para_title_text_pairs[0]
            if keep_sentence_split:
                para_title_text_pairs = {para_title_text_pairs[0]: para_title_text_pairs[1]}
            else:
                para_title_text_pairs = {para_title_text_pairs[0]: "".join(para_title_text_pairs[1])}
            context.update(para_title_text_pairs)

        return context
    
    # load sampled text included in the target articles, with two stage tfidf retrieval.
    def load_sampled_para_text_and_linked_titles(self,
                                doc_names,
                                question,
                                pruning_l, 
                                prune_after_agg=True):

        context = {}
        linked_titles = {}
        tfidf_vectorizer = TopTfIdf(n_to_select=pruning_l,
                                    filter_dist_one=True, rank=True)
        para_dict_all = {}
        linked_titles_dict_all = {}
        for doc_name in doc_names:
            paras_dict, linked_titles_dict = load_para_and_linked_titles_dict_from_tfidf_id(
                doc_name, self.db)
            if len(paras_dict) == 0:
                continue
            if prune_after_agg is True:
                para_dict_all.update(paras_dict)
                linked_titles_dict_all.update(linked_titles_dict)
            else:
                pruned_para_dict = prune_top_k_paragraphs(
                    question, paras_dict, tfidf_vectorizer, pruning_l)

                # add top pruning_l paragraphs from the target article.
                context.update(pruned_para_dict)
                # add hyperlinked paragraphs of the top pruning_l paragraphs from the target article.
                pruned_linked_titles = {k: v for k, v in linked_titles_dict.items() if k in pruned_para_dict}
                assert len(pruned_para_dict) == len(pruned_linked_titles)
                linked_titles.update(pruned_linked_titles)
                
        if prune_after_agg is True:
            pruned_para_dict = prune_top_k_paragraphs(question, para_dict_all, tfidf_vectorizer, pruning_l)
            context.update(pruned_para_dict)
            pruned_linked_titles = {
                k: v for k, v in linked_titles_dict_all.items() if k in pruned_para_dict}
            assert len(pruned_para_dict) == len(pruned_linked_titles)
            linked_titles.update(pruned_linked_titles)
            
        return context, linked_titles

    def retrieve_titles_w_tag_me(self, question, tagme_api_key):
        import tagme
        tagme.GCUBE_TOKEN = tagme_api_key
        q_annotations = tagme.annotate(question)
        tagged_titles = []
        for ann in q_annotations.get_annotations(0.1):
            tagged_titles.append(ann.entity_title)
        return tagged_titles
    
    def load_sampled_tagged_para_text(self, question, pruning_l, tagme_api_key):
        tagged_titles = self.retrieve_titles_w_tag_me(question, tagme_api_key)
        tagged_doc_names = [normalize(title) for title in tagged_titles]

        context, _ = self.load_sampled_para_text_and_linked_titles(
            tagged_doc_names, question, pruning_l)
        
        return context
    
    def get_abstract_tfidf(self,
                           q_id,
                           question,
                           args):

        doc_names, _ = self.ranker.closest_docs(question, k=args.tfidf_limit)
        # Add TFIDF close documents
        context = self.load_abstract_para_text(doc_names)

        return [{"question": question,
                 "context": context,
                 "q_id": q_id}]

    def get_article_tfidf_with_hyperlinked_titles(self, q_id,question, args):
        """
        Retrieve articles with their corresponding hyperlinked titles.
        Due to efficiency, we sample top k articles, and then sample top l paragraphs from each article. 
        (so, eventually we get k*l paragraphs with tfidf-based pruning.)
        We also store the hyperlinked titles for each paragraph. 
        """

        tfidf_limit, pruning_l, prune_after_agg = args.tfidf_limit, args.pruning_l, args.prune_after_agg
        doc_names, _ = self.ranker.closest_docs(question, k=tfidf_limit)
        context, hyper_linked_titles = self.load_sampled_para_text_and_linked_titles(
            doc_names, question, pruning_l, prune_after_agg)
        
        if args.tagme is True and args.tagme_api_key is not None:
            # if add TagMe
            tagged_context = self.load_sampled_tagged_para_text(
                question, pruning_l, args.tagme_api_key)
            
            return [{"question": question,
                     "context": context,
                     "tagged_context": tagged_context,
                     "all_linked_para_title_dic": hyper_linked_titles,
                     "q_id": q_id}]
        else:
            return [{"question": question,
                    "context": context,
                    "all_linked_para_title_dic": hyper_linked_titles,
                    "q_id": q_id}]


    def get_hyperlinked_abstract_paragraphs(self,
                                            title: str,
                                            question: str = None):

        if self.use_full_article is True and self.title2hyperlink_dic is not None:
            if title not in self.title2hyperlink_dic:
                return {}
            hyper_linked_titles = self.title2hyperlink_dic[title]
        elif self.use_full_article is True and self.title2hyperlink_dic is None:
            # for full article version, we need to store title2hyperlink_dic beforehand.
            raise NotImplementedError()
        else:
            hyper_linked_titles = self.db.get_hyper_linked(normalize(title))
        
        if hyper_linked_titles is None:
            return {}
        # if there are any hyperlinked titles, add the information to all_linked_paragraph
        all_linked_paras_dic = {}

        if self.use_full_article is True and self.title2hyperlink_dic is not None: 
            for hyper_linked_para_title in hyper_linked_titles:
                paras_dict, _ = load_para_and_linked_titles_dict_from_tfidf_id(
                    hyper_linked_para_title, self.db)
                # Sometimes article titles are updated over times but the hyperlinked titles are not. e.g., Winds <--> Wind
                # in our current database, we do not handle these "redirect" cases and thus we cannot recover.
                # If we cannot retrieve the hyperlinked articles, we just discard these articles.
                if len(paras_dict) == 0:
                    continue
                tfidf_vectorizer = TopTfIdf(n_to_select=self.pruning_l,
                                            filter_dist_one=True, rank=True)
                pruned_para_dict = prune_top_k_paragraphs(
                    question, paras_dict, tfidf_vectorizer, self.pruning_l)

                all_linked_paras_dic.update(pruned_para_dict)
        
        else:
            for hyper_linked_para_title in hyper_linked_titles:
                para_title_text_pairs = load_para_collections_from_tfidf_id_intro_only(
                    hyper_linked_para_title, self.db)
                # Sometimes article titles are updated over times but the hyperlinked titles are not. e.g., Winds <--> Wind
                # in our current database, we do not handle these "redirect" cases and thus we cannot recover.
                # If we cannot retrieve the hyperlinked articles, we just discard these articles.
                if len(para_title_text_pairs) == 0:
                    continue

                para_title_text_pairs = {para[0]: "".join(para[1])
                                        for para in para_title_text_pairs}

                all_linked_paras_dic.update(para_title_text_pairs)

        return all_linked_paras_dic
