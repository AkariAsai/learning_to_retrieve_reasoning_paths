## Reasoning Path Reader
This directory includes codes for our reasoning path reader model described in Section 3.2 of our paper.   
Our reader model is based on BERT QA model ([Devlin et al. 2019](https://arxiv.org/abs/1810.04805)), and we extend it to jointly predict answer spans and plausibility of reasoning paths selected by our retriever components. 

Table of contents:
- <a href="#1-training">1. Training</a>
- <a href="#2-evaluation">2. Evaluation</a>

## 1. Training
### Training data
We use [rc_utils.py](rc_utils.py) to train our reasoning path reader models.   
To train our reader, we first convert the original MRC datasets into SQuAD (v.2) data format, adding distant examples and negative examples.

We provide the pre-processed train and dev data files for all three datasets here (google drive):

- [HotpotQA reader train data](https://drive.google.com/file/d/1BZXSZXN99Mb7--4u0x58cixBTon1PX8N/view?usp=sharing)
- [SQuAD reader train data](https://drive.google.com/file/d/1aMTXIxYZCAC6sX5mZt6nytYxeKvjuigq/view?usp=sharing)
- [Natural Questions train data](https://drive.google.com/file/d/1wUlRkC3_yJnEzdxduFE__yQSfWa_3l0j/view?usp=sharing)


We explain some of the some required arguments below. 

- `--bert_model` <br>
This is a bert model type (e.g., `bert-base-uncased`). In our paper, we experiment both with `bert-base-uncased` and `bert-large-uncased-whole-word-masking`.

- `--output_dir` <br>
This is a directory path to save model checkpoints; a checkpoint is saved every half epoch during training.

- `--train_file` <br>
This is a file path to train data you can download from the link mentioned above. 

- `--version_2_with_negative` <br>
Please add this option to train our reader model with negative examples.

- `--do_lower_case` <br>
We use lower-cased version of BERT following previous papers in machine reading comprehension. To reproduce the results, please add this option.

There are some optional arguments; please see the full list from our [rc_utils.py](rc_utils.py). 

- `--train_batch_size` <br>
This is to specify the number of the batch size during training (default=`32`).  
*To train BERT large QA models, you are likely to reduce the number of train batch size (currently set to 32) to make it fit to your GPU memory.*

- `--max_seq_length` <br>
This is to set the maximum length of input sequence and when the input exceeds the limits, we split the data into several windows. 

- `--predict_file` <br>
This is a file path to your inference data if you would like to evaluate the reader performance (See the details below). Your `predict_file` must be in SQuAD v.2 format like `train_file`.

You can run training the command below.

```bash
python run_reader_confidence.py \
--bert_model bert-base-uncased \
--output_dir /path/to/your/output/dir \
--train_file /path/to/your/train/file \
--predict_file /path/to/your/eval/file \
--max_seq_length 384 \
--do_train \
--do_predict \
--do_lower_case \
--version_2_with_negative 
```

e.g., HotpotQA

```bash
python run_reader_confidence.py \
--bert_model bert-base-uncased \
--output_dir output_hotpot_bert_base \
--train_file data/hotpot/hotpot_reader_train_data.json \
--predict_file data/hotpot/hotpot_dev_squad_v2.0_format.json \
--max_seq_length 384 \
--do_train \
--do_predict \
--do_lower_case \
--version_2_with_negative 
```

## 2. Evaluation
As the main goal of this work is on improving open-domain QA performance, we recommend you running the pipeline to evaluate your reader performance.    
Alternatively, you can run sanity check on HotpotQA gold paragraph only settings.

#### Sanity check on HotpotQA gold only setting
For the sanity check, you can run the evaluation of the reader model performance on preprocessed dev file. 

The original HotpotQA questions contain 10 paragraphs, we discard the 8 distractor paragraphs and keep only gold paragraphs. The preprocessed data is also available [here](https://drive.google.com/open?id=1MysthH2TRYoJcK_eLOueoLeYR42T-JhB). 

You can download [the SQuAD 2.0 evaluation script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/). 

**note: the F1 calculation of is slightly different from the original HotpotQA eval script. We use the SQuAD 2.0 evaluation script for quick sanity check. Please do not use the number to report the performance on HotpotQA.**

You can run evaluation with the command below:

```bash
python evaluate-v2.0.py \
/path/to/eval/file/hotpot_dev_squad_v2.0_format.json \
/path/to/your/output/dir/predictions.json
```
The F1/EM scores of the bert-base-uncased model on the gold-paragraph only HotpotQA distractor dev data is as follows:

```py
{
  "exact": 60.60769750168805,
  "f1": 74.45707974099558,
  "total": 7405,
  "HasAns_exact": 60.60769750168805,
  "HasAns_f1": 74.45707974099558,
  "HasAns_total": 7405
}
```
