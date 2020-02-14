"""Interactive mode for the tfidf DrQA retriever module."""

import argparse
import code
import prettytable
import logging
try:
    from retriever.doc_db import DocDB
    from retriever.tfidf_doc_ranker import TfidfDocRanker
except:
    from doc_db import DocDB
    from tfidf_doc_ranker import TfidfDocRanker

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--db_save_path', type=str, default=None)
args = parser.parse_args()
db = DocDB(args.db_save_path)

logger.info('Initializing ranker...')
ranker = TfidfDocRanker(tfidf_path=args.model)


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------


def process(query, k=1, with_content=False):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    if with_content is True:
        doc_text = []
        for doc_name in doc_names:
            doc_text.append(db.get_doc_text(doc_name))
        table = prettytable.PrettyTable(
            ['Rank', 'Doc Id', 'Doc Text']
        )

        for i in range(len(doc_names)):
            table.add_row([i + 1, doc_names[i], doc_text[i]])
        print(table)

    else:
        table = prettytable.PrettyTable(
            ['Rank', 'Doc Id', 'Doc Score']
        )
        for i in range(len(doc_names)):
            table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
        print(table)


banner = """
Interactive TF-IDF DrQA Retriever
>> process(question, k=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
