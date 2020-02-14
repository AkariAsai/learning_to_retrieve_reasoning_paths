import sqlite3
import argparse
import time
try:
    from retriever.utils import find_hyper_linked_titles, remove_tags, normalize
except:
    from utils import find_hyper_linked_titles, remove_tags, normalize

class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
    
    def get_hyper_linked(self, doc_id):
        """Fetch the hyper-linked titles of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT linked_title FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if (result is None or len(result[0]) == 0) else [normalize(title) for title in result[0].split("\t")]
    
    def get_original_title(self, doc_id):
        """Fetch the original title name  of the doc."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT original_title FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
    
    def get_doc_text_hyper_linked_titles_for_articles(self, doc_id):
        """
        fetch all of the paragraphs with their corresponding hyperlink titles.
        e.g., 
        >>> paras, links = db.get_doc_text_hyper_linked_titles_for_articles("Tokyo Imperial Palace_0")
        >>> paras[2]
        'It is built on the site of the old Edo Castle. The total area including the gardens is . During the height of the 1980s Japanese property bubble, the palace grounds were valued by some to be more than the value of all of the real estate in the state of California.'
        >>> links[2]
        ['Edo Castle', 'Japanese asset price bubble', 'Real estate', 'California']
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        if result is None:
            return [], []
        else:
            hyper_linked_paragraphs = result[0].split("\n\n")
            paragraphs, hyper_linked_titles = [], []
            
            for hyper_linked_paragraph in hyper_linked_paragraphs:
                paragraphs.append(remove_tags(hyper_linked_paragraph))
                hyper_linked_titles.append([normalize(title) for title in find_hyper_linked_titles(
                    hyper_linked_paragraph)])
                
            return paragraphs, hyper_linked_titles
