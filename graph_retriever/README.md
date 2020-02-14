# Graph-based Recurrent Retriever

This directory includes codes for our graph-based recurrent retriever model described in Section 3.1.1 of our paper.

Table of contents:
- <a href="#1-training">1. Training</a>
- <a href="#2-inference">2. Inference</a> (optional)

## 1. Training
We use `run_graph_retriever.py` to train the model.
We first need to prepare training data for each task; in our paper, we ued HotpotQA, HotpotQA distractor, Natural Questions Open, and SQuAD Open.
To train the model with the same settings used in our paper, we explain some of the most important arguments below.
See each example for more details about the values of the arguments.

<b>Note: it is not possible to perfectly reproduce the experimental results even with exactly the same settings, due to device or environmental differences.
When we trained the model for the SQuAD Open dataset five times with five different random seeds, the standard deviation of the final QA EM score was around `0.5`.
Therefore, you would occasionally face a different score by ~1% when changing random seeds or something.</b>
In our paper, we reported all the results based on the default seed (42) in the BERT code base.

- `--task` <br>
This is a required argument to specify which dataset we use.
Currently, the valid values are `hotpot_open`, `hotpot_distractor`, `squad`, and `nq`.

- `--train_file_path` <br>
This has to be specified for training.
This can be either a singe file or a set of split files; for the latter case, we simply use the `glob` package to read all the files associated with a partial file path.
For example, if we have three files named `./data_1.json`, `./data_2.json`, and `./data_3.json`, then `--train_file_path ./data_` allows you to load all the three files by `glob.glob(./data_*)`.
This is useful when the single data file is too big and we want to split it into smaller files.

- `--output_dir` <br>
This is a directory path to save model checkpoints; a checkpoint is saved every half epoch during training.
The checkpoint files are `pytorch_model_0.5.bin`, `pytorch_model_1.bin`, `pytorch_model_1.5.bin`, etc.

- `--max_para_num` <br>
This is the number of paragraphs associated with a question.
If `--max_para_num` is `N` and the number of the ground-truth paragraphs is `2` for the question, then there are `N-2` paragraphs as negative examples for training.
We expect higher accuracy with larger values of `N`, but there is a trade-off with the training time.

- `--tfidf_limit` <br>
This is specifically used for HotpotQA, where we use negative examples from both TF-IDF-based and hyperlink-based paragraphs.
If `--max_para_num` is `N` and `--tfidf_limit` is `M` (`N` >= `M`), then there are `M` TF-IDF-based negative examples and `N-M` hyperlink-based negative examples.

- `--neg_chunk` <br>
This is used to control GPU memory consumption.
Our model training needs to handle many paragraphs for a question with BERT, so it is not feasible to run forward/backward functions all together.
To resolve this issue, we have this argument to split the negative examples into small chunks, where the chunk size can be specified with this argument.
We used NVIDIA V100 GPUs (with 16GB memory) for our experiments, and `--neg_chunk 8` works.
For other GPUs with less memory, please consider using smaller number for this argument.
It should be noted that, changing this value does not affect the results; our model does not use the softmax normalization, and thus we can run the forward/backward functions separately for each chunk.

- `--train_batch_size` & `--gradient_accumulation_steps` <br>
These control the mini-batch size and how often we update the model parameters with the optimizer.
More importantly, these depend on how many GPUs we can use.
Due to the model size of BERT and the number of paragraphs to be processed, one GPU can handle one example.
That means, `--train_batch_size 4` and `--gradient_accumulation_steps 4` work on a single GPU, but `--train_batch_size 4` and `--gradient_accumulation_steps 1` do not work due to OOM.
However, if we have four GPUs, for example, the latter setting works because the four examples can be handled by the four GPUs.

- `--use_redundant` and `--use_multiple_redundant` <br>
These are used to use the data augmentation technique for the sake of robustness.
`--use_redundant` allows you to use one additional training example for a question, and `--use_multiple_redundant` allows you to use multiple examepls.
To further specify how many examples can be used for the training, you can specify `--max_redundant_num`.

- `max_select_num` <br>
This is set to specify the maximum number of reasoning steps in our model.
This value should be `K+1`, where `K` is the number of ground-truth paragraphs, and `1` is for the EOE symbol.
You further need to add `1` when using the `--use_redundant` option.
For example, for HotpotQA, `K` is 2 and we used the `--use_redundant` option, and then the total value is `4`.

- `--example_limit` <br>
This allows you to sanity-check your running the code, by limiting the number of examples (i.e., questions) to load from each file.
This is useful in checking if everything goes well in your environments.

### HotpotQA
For HotpotQA, we used up tp 50 paragraphs for each question, and among them up to 40 paragraphs are from the TF-IDF retriever.
The other 10 paragraphs are from hyperlinks.
The following command assumes the use of four V100 GPUs, but with other devices, you might modify `--neg_chunk` and `--gradient_accumulation_steps`.

```bash
python run_graph_retriever.py \
--task hotpot_open \
--bert_model bert-base-uncased --do_lower_case \
--train_file_path <file path to the training data> \
--output_dir <directory to save model checkpoints> \
--max_para_num 50 \
--tfidf_limit 40 \
--neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 \
--learning_rate 3e-5 --num_train_epochs 3 \
--use_redundant \
--max_select_num 4 \
```

You can use the files in `hotpotqa_new_selector_train_data_db_2017_10_12_fix.zip` for `--train_file_path`.

### HotpotQA distractor
For HotpotQA distractor, `--max_para_num` is always 10, due to the task setting.
The following command assumes the use of one V100 GPU, but with other devices, you might modify `--neg_chunk`.

```bash
python run_graph_retriever.py \
--task hotpot_distractor \
--bert_model bert-base-uncased --do_lower_case \
--train_file_path <file path to the training data> \
--output_dir <directory to save model checkpoints> \
--max_para_num 10 \
--neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 4 \
--learning_rate 3e-5 --num_train_epochs 3 \
--max_select_num 3
```

You can use `hotpot_train_order_sensitive.json` for `--train_file_path`.

### Natural Questions Open
For Natural Questions, we used up to 80 paragraphs for each question; changing this to 50 or so does not make significant difference, but in general, the more negative examples, the better (at least, not worse).
To encourage the multi-step nature, we used the `--use_multiple_redundant` option with a larger mini-batch size, because the number of training examples is significantly increased.
The following command assumes the use of four V100 GPUs, but with other devices, you might modify `--neg_chunk` and `--gradient_accumulation_steps`.

```bash
python run_graph_retriever.py \
--task nq \
--bert_model bert-base-uncased --do_lower_case \
--train_file_path <file path to the training data> \
--output_dir <directory to save model checkpoints> \
--max_para_num 80 \
--neg_chunk 8 --train_batch_size 8 --gradient_accumulation_steps 2 \
--learning_rate 2e-5 --num_train_epochs 3 \
--use_multiple_redundant \
--max_select_num 3 \
```

You can use the files in `nq_selector_train.tar.gz` for `--train_file_path`.

### SQuAD Open
For SQuAD, we used up to 50 paragraphs for each question.
The following command assumes the use of four V100 GPUs, but with other devices, you might modify `--neg_chunk` and `--gradient_accumulation_steps`.

```bash
python run_graph_retriever.py \
--task squad \
--bert_model bert-base-uncased --do_lower_case \
--train_file_path <file path to the training data> \
--output_dir <directory to save model checkpoints> \
--max_para_num 50 \
--neg_chunk 8 --train_batch_size 4 --gradient_accumulation_steps 1 \
--learning_rate 2e-5 --num_train_epochs 3 \
--max_select_num 2
```

You can use `squad_tfidf_rgs_train_tfidf_top_negative_example.json` for `--train_file_path`.

## 2. Inference
The trained models can be evaluated in our pipelined evaluation script.
However, we can also use `run_graph_retriever.py` to run trained models for inference.
This is in particular useful in sanity checking retrieval accuracy for HotpotQA distractor.
To run the models with the same settings used in our paper, we explain some of the most important arguments below.
Some of th arguments are also used in the pipelined evaluation.

- `--dev_file_path` <br>
This has to be specified for evaluation or inference.
The semantics is the same as `--train_file_path` in that you can use either a single file or a set of multiple files.

- `--pred_file` <br>
This has to be specified for evaluation or inference.
This is a file path to save the model's prediction results to be used for the next reading step.
Note that, if you used split files for `--dev_file_path`, you may need to merge the output json files later.

- `--output_dir` and `--model_suffix` <br>
This is based on `--output_dir` for training.
To load a trained model from `pytorch_model_1.5.bin`, you need to set `--model_suffix 1.5`.

- `--max_para_num` <br>
This is used to specify the maximum number of paragraphs, including hyper-linked ones, for each question.
This is typically set to a large number to cover all the possible paragraphs.

- `--beam` <br>
This is used to specify a beam size for our beam search algorithm to retrieve reasoning paths.

- `--pruning_by_links` <br>
This option is used to do pruning during the beam search, based on hyper-links.

- `--exapnd_links` <br>
This options is used to add within-document links, along with the default hyper-links on Wikipedia.

- `--no_links` <br>
This option is used to avoid using the link information, to see how effective the use of the links is.

- `--tagme` <br>
This is used to add TagMe-based paragraphs for each question, for better initial retrieval.

- `--eval_chunk` <br>
This option's purpose is similar to that of `--neg_chunk`.
If an evaluation file is too big, it would not fit in CPU RAM, and by this option we can specify a chunk size to run evaluation by avoiding processing all the evaluation examples together.

- `--split_chunk` <br>
This is useful to control the use of GPU RAM for the BERT encoding.
This is the number of paragraphs to be encoded by BERT together.
The smaller the value is, the less GPU memory is consumed.

### HotpotQA

```bash
python run_graph_retriever.py \
--task hotpot_open \
--bert_model bert-base-uncased --do_lower_case \
--dev_file_path <file path to the evaluation data> \
--pred_file <file path to output the prediction results> \
--output_dir <directory used for training> \
--model_suffix 2 \
--max_para_num 2500 \
--beam 8 \
--pruning_by_links \
--eval_chunk 500 \
--split_chunk 300
```

### HotpotQA distractor

```bash
python run_graph_retriever.py \
--task hotpot_distractor \
--bert_model bert-base-uncased --do_lower_case \
--dev_file_path <file path to the evaluation data> \
--pred_file <file path to output the prediction results> \
--output_dir <directory used for training> \
--model_suffix 2 \
--max_para_num 10 \
--beam 8 \
--pruning_by_links \
--eval_chunk 500 \
--split_chunk 300
```

You can use `hotpot_fake_sq_dev_new.json` for `--train_file_path`.

### Natural Questions Open

```bash
python run_graph_retriever.py \
--task nq \
--bert_model bert-base-uncased --do_lower_case \
--dev_file_path <file path to the evaluation data> \
--pred_file <file path to output the prediction results> \
--output_dir <directory used for training> \
--model_suffix 2 \
--max_para_num 2000 \
--beam 8 \
--pruning_by_links \
--expand_links \
--tagme \
--eval_chunk 500 \
--split_chunk 300
```

### SQuAD Open

```bash
python run_graph_retriever.py \
--task squad \
--bert_model bert-base-uncased --do_lower_case \
--dev_file_path <file path to the evaluation data> \
--pred_file <file path to output the prediction results> \
--output_dir <directory used for training> \
--model_suffix 2 \
--max_para_num 500 \
--beam 8 \
--no_links \
--eval_chunk 500 \
--split_chunk 300
```
