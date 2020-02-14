import unicodedata
import jsonlines
import re
from urllib.parse import unquote
import regex
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize(text):
    """Resolve different type of unicode encodings / capitarization in HotpotQA data."""
    text = unicodedata.normalize('NFD', text)
    return text[0].capitalize() + text[1:]


def make_wiki_id(title, para_index):
    title_id = "{0}_{1}".format(normalize(title), para_index)
    return title_id


def find_hyper_linked_titles(text_w_links):
    titles = re.findall(r'href=[\'"]?([^\'" >]+)', text_w_links)
    titles = [unquote(title) for title in titles]
    titles = [title[0].capitalize() + title[1:] for title in titles]
    return titles


TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def process_jsonlines(filename):
    """
    This is process_jsonlines method for extracted Wikipedia file.
    After extracting items by using Wikiextractor (with `--json` and `--links` options), 
    you will get the files named with wiki_xx, where each line contains the information of each article.
    e.g., 
    {"id": "316", "url": "https://en.wikipedia.org/wiki?curid=316", "title": "Academy Award for Best Production Design", 
    "text": "Academy Award for Best Production Design\n\nThe <a href=\"Academy%20Awards\">Academy Award</a> for 
    Best Production Design recognizes achievement for <a href=\"art%20direction\">art direction</a> \n\n"}
    This function takes these input and extract items.
    Each article contains one or more than one paragraphs, and each paragraphs are separeated by \n\n.
    """
    # item should be nested list
    extracted_items = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            wiki_id = obj["id"]
            title = obj["title"]
            title_id = make_wiki_id(title, 0)
            text_with_links = obj["text"]

            hyper_linked_titles_text = ""
            # When we consider the whole article as a document unit (e.g., SQuAD Open, Natural Questions Open)
            # we'll keep the links with the original articles, and dynamically process and extract the links
            # when we process with our selector.
            extracted_items.append({"wiki_id": wiki_id, "title": title_id,
                                    "plain_text": text_with_links,
                                    "hyper_linked_titles": hyper_linked_titles_text,
                                    "original_title": title})

    return extracted_items

def process_jsonlines_hotpotqa(filename):
    """
    This is process_jsonlines method for intro-only processed_wikipedia file.
    The item example:
    {"id": "45668011", "url": "https://en.wikipedia.org/wiki?curid=45668011", "title": "Flouch Roundabout",
     "text": ["Flouch Roundabout is a roundabout near Penistone, South Yorkshire, England, where the A628 meets the A616."],
     "charoffset": [[[0, 6],...]]
     "text_with_links" : ["Flouch Roundabout is a roundabout near <a href=\"Penistone\">Penistone</a>,
     <a href=\"South%20Yorkshire\">South Yorkshire</a>, England, where the <a href=\"A628%20road\">A628</a>
     meets the <a href=\"A616%20road\">A616</a>."],
        "charoffset_with_links": [[[0, 6], ... [213, 214]]]}
    """
    # item should be nested list
    extracted_items = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            wiki_id = obj["id"]
            title = obj["title"]
            title_id = make_wiki_id(title, 0)
            plain_text = "\t".join(obj["text"])
            text_with_links = "\t".join(obj["text_with_links"])

            hyper_linked_titles = []
            hyper_linked_titles = find_hyper_linked_titles(text_with_links)
            if len(hyper_linked_titles) > 0:
                hyper_linked_titles_text = "\t".join(hyper_linked_titles)
            else:
                hyper_linked_titles_text = ""
            extracted_items.append({"wiki_id": wiki_id, "title": title_id,
                                    "plain_text": plain_text,
                                    "hyper_linked_titles": hyper_linked_titles_text,
                                    "original_title": title})

    return extracted_items


# ------------------------------------------------------------------------------
# Sparse matrix saving/loading helpers.
# ------------------------------------------------------------------------------


def save_sparse_csr(filename, matrix, metadata=None):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)


def load_sparse_csr(filename):
    loader = np.load(filename, allow_pickle=True)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None

# ------------------------------------------------------------------------------
# Token hashing.
# ------------------------------------------------------------------------------


def hash(token, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    return murmurhash3_32(token, positive=True) % num_buckets

# ------------------------------------------------------------------------------
# Text cleaning.
# ------------------------------------------------------------------------------


STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}


def filter_word(text):
    """Take out english stopwords, punctuation, and compound endings."""
    text = normalize(text)
    if regex.match(r'^\p{P}+$', text):
        return True
    if text.lower() in STOPWORDS:
        return True
    return False


def filter_ngram(gram, mode='any'):
    """Decide whether to keep or discard an n-gram.
    Args:
        gram: list of tokens (length N)
        mode: Option to throw out ngram if
          'any': any single token passes filter_word
          'all': all tokens pass filter_word
          'ends': book-ended by filterable tokens
    """
    filtered = [filter_word(w) for w in gram]
    if mode == 'any':
        return any(filtered)
    elif mode == 'all':
        return all(filtered)
    elif mode == 'ends':
        return filtered[0] or filtered[-1]
    else:
        raise ValueError('Invalid mode: %s' % mode)


def get_field(d, field_list):
    """get the subfield associated to a list of elastic fields
        E.g. ['file', 'filename'] to d['file']['filename']
    """
    if isinstance(field_list, str):
        return d[field_list]
    else:
        idx = d.copy()
        for field in field_list:
            idx = idx[field]
        return idx


def load_para_collections_from_tfidf_id_intro_only(tfidf_id, db):
    if "_0" not in tfidf_id:
        tfidf_id = "{0}_0".format(tfidf_id)
    if db.get_doc_text(tfidf_id) is None:
        logger.warning("{0} is missing".format(tfidf_id))
        return []
    return [[tfidf_id, db.get_doc_text(tfidf_id).split("\t")]]

def load_linked_titles_from_tfidf_id(tfidf_id, db):
    para_titles = db.get_paras_with_article(tfidf_id)
    linked_titles_all = []
    for para_title in para_titles:
        linked_title_per_para = db.get_hyper_linked(para_title)
        if len(linked_title_per_para) > 0:
            linked_titles_all += linked_title_per_para.split("\t")
    return linked_titles_all

def load_para_and_linked_titles_dict_from_tfidf_id(tfidf_id, db):
    """
    load paragraphs and hyperlinked titles from DB. 
    This method is mainly for Natural Questions Open benchmark.
    """
    # will be fixed in the later version; current tfidf weights use indexed titles as keys.
    if "_0" not in tfidf_id:
        tfidf_id = "{0}_0".format(tfidf_id)
    paras, linked_titles = db.get_doc_text_hyper_linked_titles_for_articles(
        tfidf_id)
    if len(paras) == 0:
        logger.warning("{0} is missing".format(tfidf_id))
        return [], []

    paras_dict = {}
    linked_titles_dict = {}
    article_name = tfidf_id.split("_0")[0]
    # store the para_dict and linked_titles_dict; skip the first para (title)
    for para_idx, (para, linked_title_list) in enumerate(zip(paras[1:], linked_titles[1:])):
        paras_dict["{0}_{1}".format(article_name, para_idx)] = para
        linked_titles_dict["{0}_{1}".format(
            article_name, para_idx)] = linked_title_list

    return paras_dict, linked_titles_dict

def prune_top_k_paragraphs(question_text, paragraphs, tfidf_vectorizer, pruning_l=10):
    para_titles, para_text = list(paragraphs.keys()), list(paragraphs.values())
    # prune top l paragraphs using the question as query to reduce the search space.
    top_tfidf_para_indices = tfidf_vectorizer.prune(
        question_text, para_text)[:pruning_l]
    para_title_text_pairs_pruned = {}

    # store the selected paras into dictionary.
    for idx in top_tfidf_para_indices:
        para_title_text_pairs_pruned[para_titles[idx]] = para_text[idx]

    return para_title_text_pairs_pruned
