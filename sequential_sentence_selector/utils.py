from __future__ import absolute_import, division, print_function

import os

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertModel, BertTokenizer)

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
}

def get_bert_model_from_pytorch_transformers(model_name):
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
    config = config_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, from_tf=bool('.ckpt' in model_name), config=config)

    tokenizer = tokenizer_class.from_pretrained(model_name)
    
    vocab_file_name = './vocabulary_'+model_name+'.txt'

    if not os.path.exists(vocab_file_name):
        index = 0
        with open(vocab_file_name, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(tokenizer.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    assert False
                    index = token_index
                writer.write(token + u'\n')
                index += 1

    return model.state_dict(), vocab_file_name
