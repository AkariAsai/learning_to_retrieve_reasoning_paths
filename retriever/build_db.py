#!/usr/bin/env python3
# The codes are started from DrQA (https://github.com/facebookresearch/DrQA) library.
"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util
import glob

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm

try:
    from retriever.utils import normalize, process_jsonlines_hotpotqa, process_jsonlines
except:
    from utils import normalize, process_jsonlines_hotpotqa, process_jsonlines
    
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


PREPROCESS_FN = None


def init(filename):
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents_hotpotqa(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []
    extracted_items = process_jsonlines_hotpotqa(filename)
    for extracted_item in extracted_items:
        title = extracted_item["title"]
        text = extracted_item["plain_text"]
        original_title = extracted_item["original_title"]
        hyper_linked_titles = extracted_item["hyper_linked_titles"]

        documents.append((title, text,
                          hyper_linked_titles, original_title))
    return documents

def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []
    extracted_items = process_jsonlines(filename)
    for extracted_item in extracted_items:
        title = extracted_item["title"]
        text = extracted_item["plain_text"]
        original_title = extracted_item["original_title"]
        hyper_linked_titles = extracted_item["hyper_linked_titles"]

        documents.append((title, text,
                          hyper_linked_titles, original_title))
    return documents

def store_contents(wiki_dir, save_path, preprocess, num_workers=None, hotpotqa_format=False):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        preprocess: Path to file defining a custom `preprocess` function. Takes
          in and outputs a structured doc.
        num_workers: Number of parallel processes to use when reading docs.
    """
    filenames = [f for f in glob.glob(
        wiki_dir + "/*/wiki_*", recursive=True) if ".bz2" not in f]
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE documents (id PRIMARY KEY, text, linked_title, original_title);")

    workers = ProcessPool(num_workers, initializer=init,
                          initargs=(preprocess,))
    count = 0
    # Due to the slight difference of input format between preprocessed HotpotQA wikipedia data and 
    # the ones by Wikiextractor, we call different functions for data process.
    if hotpotqa_format is True:
        content_processing_method = get_contents_hotpotqa
    else:
        content_processing_method = get_contents
        
    with tqdm(total=len(filenames)) as pbar:
        for pairs in tqdm(workers.imap_unordered(content_processing_method, filenames)):
            count += len(pairs)
            c.executemany(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?)", pairs)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wiki_dir', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--preprocess', type=str, default=None,
                        help=('File path to a python module that defines '
                              'a `preprocess` function'))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--hotpoqa_format', action='store_true',
                        help='the input files are hotpotqa format.')
    args = parser.parse_args()

    store_contents(
        args.wiki_dir, args.save_path, args.preprocess, args.num_workers,
        args.hotpoqa_format
    )
