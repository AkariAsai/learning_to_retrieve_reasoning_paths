# TF-IDF based retriever
For efficiency and scalability, we bootstrap retrieval process with a TF-IDF based retrieval. In particular, we first sample seed paragraphs based on TF-IDF scores and sequentially retrieve sub-graphs connected from the seed paragraphs. 

Table of contents:
- <a href="#1-preprocessing-wikipedia-dump">1. Preprocessing Wikipedia dump</a>
- <a href="#2-building-database">2. Building database</a>
- <a href="#3-building-the-tf-idf-n-grams">3. Building the TF-IDF N-grams</a>
- <a href="#4-interactive-mode">4. Interactive mode</a>

*Acknowledgement: The code bases are started from the amazing [DrQA](https://github.com/facebookresearch/DrQA)'s document retriever code. We really appreciate the efforts by the DrQA authors.*

## 1. Preprocessing Wikipedia dump

#### download Wikipedia dumps
First, you need to install the Wikipedia dump. 

- **HotpotQA**: You do not need to download Wikipedia dump by yourself, as the authors' provide the preprocessed dump in [HotpotQA official website](https://hotpotqa.github.io/wiki-readme.html). If you consider using our model for HotpotQA only, we recommend you downloading the [intro-paragraph only version](https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2). You can extract files as the instruction in the website. 

- **SQuAD**: Although you can use the same dump as in HotpotQA, we recommend you using the older dump and the DB distributed by the [DrQA repository](https://github.com/facebookresearch/DrQA/blob/master/download.sh). Wikipedia is frequently edited by users, and thus if you use the newer version, some answers (originally included in the context) are lost. Please refer the details of our finding in **Appendix B.5** in our paper. Although the DB does not preserve hyperlink information, we do not observe large performance information on SQuAD without link-based hop, as the questions are mostly single-hop or inner-article multi-hop.

- **Natural Questions**: For Natural Questions, due to the similar reasons mentioned above, we recommend you using the [dump](https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-meta-current.xml.bz2) from December, 2018, which also used in previous work on NQ Open ([Lee et al., 2019](https://arxiv.org/abs/1906.00300); [Min et al., 2019](https://arxiv.org/abs/1909.04849)). 


#### Extract articles
As mentioned, you do not need to preprocess the dump by yourself for HotpotQA or SQuAD. If you attempt to experiment on Natural Questions or other Wikipedia dump by yourself, you need to extract the articles. 

1. Install [wikiextractor](https://github.com/attardi/wikiextractor)
2. Run `Wikiextractor.py` with `--json` and `-l`. The first option make the output easy-to-read-and-process `jsonlines` format and the second option preserve the hyperlinks, which is crutial for our framework. 


## 2. Building database

To efficiently store and access our documents, we store them in a sqlite database.    
To create a sqlite db from a corpus of documents, run:

```bash
python build_db.py /path/to/data /path/to/saved/db.db --hotpoqa_format
```

**Note**
Do not forget `--hotpoqa_format` option when you process Wikipedia data for HotpotQA experiments. The HotpotQA authors kindly provide the preprocessed dump, and the titles and sentence separations should be consistent for supporting fact evaluations.

For introductory paragraph only Wikipedia, the total number of the paragraphs stored into DB should be 5,233,329.

```
$ python build_db.py $PATH_TO_WIKI_DIR/enwiki-20171001-pages-meta-current-withlinks-abstracts enwiki_intro.db --intro_only
07/01/2019 11:31:51 PM: [ Reading into database... ]
100%|███████████████████████████████████████| 15517/15517 [01:47<00:00, 143.97it/s]
07/01/2019 11:33:39 PM: [ Read 5233329 docs. ]
07/01/2019 11:33:39 PM: [ Committing... ]
```

If you create the DB from the 2018/12/20 dump, the total number of articles will be 5,771,730.
```
$ python retriever/build_db.py $PATH_WIKI_DIR enwiki_20181220_all.db
02/01/2020 11:52:28 PM: [ Reading into database... ]
100%|██████████████████████████████████████| 16399/16399 [04:03<00:00, 67.28it/s]
02/01/2020 11:56:32 PM: [ Read 5771730 docs. ]
02/01/2020 11:56:32 PM: [ Committing... ]
```

**Note: the total number of paragraphs would be 30M and te DB size would be 27 GB.)**

Optional arguments:
```
--preprocess    File path to a python module that defines a `preprocess` function.
--num-workers   Number of CPU processes (for tokenizing, etc).
```

#### Keeping hyperlinks in `doc_text`
Due to the nature of Wikipedia hyperlinks, a hyperlink connection is from an paragraph (source paragraph) to an article (target articles), although if we only consider introductory paragraphs, the relations are always paragraph-paragraph. 

To efficiently store the relationship, for the multiple paragraph settings (e.g., Natural Questions Open), we keep the hyperlink information in the `doc_text`. 

e.g., Seattle
```
Seattle is a <a href="port">seaport</a> city on the <a href="West%20Coast%20of%20the%20United%20States">West Coast of the United States</a>. 
```

## 3. Building the TF-IDF N-grams

To build a TF-IDF weighted word-doc sparse matrix from the documents stored in the sqlite db, run:

```bash
python build_tfidf.py /path/to/doc/db.db /path/to/output/dir
```

e.g., 
```bash
python build_tfidf.py enwiki_intro.db tfidf_results_from_enwiki_intro_only/
```

The sparse matrix and its associated metadata will be saved to the output directory under (i.e., `tfidf_results_from_enwiki_intro_only`) `<db-name>-tfidf-ngram=<N>-hash=<N>-tokenizer=<T>.npz`.


Optional arguments:
```
--ngram         Use up to N-size n-grams (e.g. 2 = unigrams + bigrams). By default only ngrams without stopwords or punctuation are kept.
--hash-size     Number of buckets to use for hashing ngrams.
--tokenizer     String option specifying tokenizer type to use (e.g. 'corenlp').
--num-workers   Number of CPU processes (for tokenizing, etc).
```

**Note: If you build TFIDF matrix from the full Wikipedia paragraphs, it ends up consuming more than a lot of CPU memories, which your local machine might not accommodate. In that case, please use the DB & .npz files we distribute, our consider using a amchine with more memory.**


## 4. Interactive mode
You can play with the TFIDF retriever with interactive mode :)   
If you set the `with_content=True` in the process function, you can see the paragraph as well as title.

```bash
python scripts/retriever/interactive.py --model /path/to/model \
--db_save_path /path/to/db file
```
e.g.,

```bash
python interactive.py --model tfidf_wiki_abst/wiki_open_full_new_db_intro_only-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz \
--db_save_path wiki_open_full_new_db_intro_only.db
```
```
>>> process('At what university can the building that served as the fictional household that includes Gomez and Morticia be found?', k=1, with_content=True)
+------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Rank |        Doc Id       |                                                                                                                                                                   Doc Text                                                                                                                                                                   |
+------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|  1   | The Addams Family_0 | The Addams Family is a fictional household created by American cartoonist Charles Addams. The Addams Family characters have traditionally included Gomez and Morticia Addams, their children Wednesday and Pugsley, close family members Uncle Fester and Grandmama, their butler Lurch, the disembodied hand Thing, and Gomez's Cousin Itt. |
+------+---------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```
