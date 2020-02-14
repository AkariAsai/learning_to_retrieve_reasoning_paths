# Sequential Sentence Selector

This directory includes codes for our sequential sentence selector model described in Appendix A.4 of our paper.
The model is the same as our proposed graph retriever model to retrieve reasoning paths, and it is adapted to the supporting fact prediction task in HotpotQA.

## Training
We use `run_sequential_sentence_selector.py` to train the model.
Basically, the overall code is based on that of our graph retriever.
Here is an example command used for our paper.
Since we used `pytorch-pretrained-bsed` to develop our models, we cannot directly use the BERT-whole-word-masking configurations.
However, we have a way to use, for example, `bert-large-uncased-whole-word-masking` when we have `pytorch-transformers` in our env.
Refer to `./utils.py` for this trick.

```bash
python run_sequential_sentence_selector.py \
--bert_model bert-large-uncased-whole-word-masking \
--train_file_path <file path to the training data> \
--output_dir <directory to save model checkpoints> \
--do_lower_case \
--train_batch_size 12 \
--gradient_accumulation_steps 1 \
--num_train_epochs 3 \
--learning_rate 3e-5
```

Once you train your model, you can use it in our evaluation pipeline script for HotpotQA.
You can use `hotpot_sf_selector_order_train.json` for `--train_file_path`.
