#!/usr/bin/env python3
# The codes are based on DrQA (https://github.com/facebookresearch/DrQA) library.
"""A script to build the tf-idf document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

try:
    from retriever.tfidf_doc_ranker import TfidfDocRanker
    from retriever.doc_db import DocDB
    from retriever.tokenizers import SimpleTokenizer
    from retriever.utils import filter_ngram, hash, save_sparse_csr
except: 
    from tfidf_doc_ranker import TfidfDocRanker
    from doc_db import DocDB
    from tokenizers import SimpleTokenizer
    from utils import filter_ngram, hash, save_sparse_csr
    
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Set retriever class
# ------------------------------------------------------------------------------


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sqlite':
        return DocDB
    raise RuntimeError('Invalid retriever class: %s' % name)

# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------


DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def fetch_text_multi_para(doc_id):
    global PROCESS_DB
    paras, _ = PROCESS_DB.get_doc_text_hyper_linked_titles_for_articles(doc_id)
    if len(paras) > 0:
        paras = "\n\n".join(paras)
    return paras


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------

def count(ngram, hash_size, multi_para, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    # FIXME: remove hard coding.
    row, col, data = [], [], []
    # Tokenize
    
    if multi_para is True:
        # 1. if multi_para is true, the doc contains multiple paragraphs separated by \n\n and with links.
        tokens = tokenize(fetch_text_multi_para(doc_id))
    else:
        # 2. if not, only intro docs are retrieved and the sentences are separated by \t.
        # remove sentence separations ("\t") (only for HotpotQA).
        tokens = tokenize(fetch_text(doc_id).replace("\t", ""))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=ngram, uncased=True, filter_fn=filter_ngram
    )

    # Hash ngrams and count occurences
    counts = Counter([hash(gram, hash_size)
                      for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(args, db, db_opts):
    """Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    db_class = get_class(db)
    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Setup worker pool
    # TODO: Add tokenizer's choice.
    tok_class = SimpleTokenizer
    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=(tok_class, db_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size, args.multi_para)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' %
                    (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids))
    )
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_ids)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix(cnts):
    """Convert the word count matrix into tfidf one.

    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Ns = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=None,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--multi_para', action='store_true',
                        help='set true if the db contains multiple paragraphs, not intro-paragraph only.')
    args = parser.parse_args()

    logging.info('Counting words...')
    count_matrix, doc_dict = get_count_matrix(
        args, 'sqlite', {'db_path': args.db_path}
    )

    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-tfidf-ngram=%d-hash=%d-tokenizer=%s' %
                 (args.ngram, args.hash_size, args.tokenizer))
    
    # check if output_dir exists; if not, create the output_dir.
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict
    }
    save_sparse_csr(filename, tfidf, metadata)
