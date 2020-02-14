from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

try:
    from graph_retriever.utils import tokenize_question
    from graph_retriever.utils import tokenize_paragraph
    from graph_retriever.utils import expand_links
except:
    from utils import tokenize_question
    from utils import tokenize_paragraph
    from utils import expand_links
    
class BertForGraphRetriever(BertPreTrainedModel):

    def __init__(self, config, graph_retriever_config):
        super(BertForGraphRetriever, self).__init__(config)

        self.graph_retriever_config = graph_retriever_config

        self.bert = BertModel(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initial state
        self.s = Parameter(torch.FloatTensor(config.hidden_size).uniform_(-0.1, 0.1))

        # Scaling factor for weight norm
        self.g = Parameter(torch.FloatTensor(1).fill_(1.0))

        # RNN weight
        self.rw = nn.Linear(2*config.hidden_size, config.hidden_size)
        
        # EOE and output bias
        self.eos = Parameter(torch.FloatTensor(config.hidden_size).uniform_(-0.1, 0.1))
        self.bias = Parameter(torch.FloatTensor(1).zero_())

        self.apply(self.init_bert_weights)
        self.cpu = torch.device('cpu')
        
    '''
    state: (B, 1, D)
    '''
    def weight_norm(self, state):
        state = state / state.norm(dim = 2).unsqueeze(2)
        state = self.g * state
        return state

    '''
    input_ids, token_type_ids, attention_mask: (B, N, L)
    B: batch size
    N: maximum number of Q-P pairs
    L: maximum number of input tokens
    '''
    def encode(self, input_ids, token_type_ids, attention_mask, split_chunk = None):
        B = input_ids.size(0)
        N = input_ids.size(1)
        L = input_ids.size(2)
        input_ids = input_ids.contiguous().view(B*N, L)
        token_type_ids = token_type_ids.contiguous().view(B*N, L)
        attention_mask = attention_mask.contiguous().view(B*N, L)

        # [CLS] vectors for Q-P pairs
        if split_chunk is None:
            encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            pooled_output = encoded_layers[:, 0]

        # an option to reduce GPU memory consumption at eval time, by splitting all the Q-P pairs into smaller chunks
        else:
            assert type(split_chunk) == int
            
            TOTAL = input_ids.size(0)
            start = 0

            while start < TOTAL:
                end = min(start+split_chunk-1, TOTAL-1)
                chunk_len = end-start+1

                input_ids_ = input_ids[start:start+chunk_len, :]
                token_type_ids_ = token_type_ids[start:start+chunk_len, :]
                attention_mask_ = attention_mask[start:start+chunk_len, :]

                encoded_layers, pooled_output_ = self.bert(input_ids_, token_type_ids_, attention_mask_, output_all_encoded_layers=False)
                encoded_layers = encoded_layers[:, 0]
                
                if start == 0:
                    pooled_output = encoded_layers
                else:
                    pooled_output = torch.cat((pooled_output, encoded_layers), dim = 0)

                start = end+1

            pooled_output = pooled_output.contiguous()
                    
        paragraphs = pooled_output.view(pooled_output.size(0)//N, N, pooled_output.size(1)) # (B, N, D), D: BERT dim
        EOE = self.eos.unsqueeze(0).unsqueeze(0) # (1, 1, D)
        EOE = EOE.expand(paragraphs.size(0), EOE.size(1), EOE.size(2)) # (B, 1, D)
        EOE = self.bert.encoder.layer[-1].output.LayerNorm(EOE)
        paragraphs = torch.cat((paragraphs, EOE), dim = 1) # (B, N+1, D)

        # Initial state
        state = self.s.expand(paragraphs.size(0), 1, self.s.size(0))
        state = self.weight_norm(state)

        return paragraphs, state
        
    '''
    input_ids, token_type_ids, attention_mask: (B, N, L)
    - B: batch size
    - N: maximum number of Q-P pairs
    - L: maximum number of input tokens

    output_mask, target: (B, max_num_steps, N+1)
    '''
    def forward(self, input_ids, token_type_ids, attention_mask, output_mask, target, max_num_steps):

        paragraphs, state = self.encode(input_ids, token_type_ids, attention_mask)

        for i in range(max_num_steps):
            if i == 0:
                h = state
            else:
                input = paragraphs[:, i-1:i, :] # (B, 1, D)
                state = torch.cat((state, input), dim = 2) # (B, 1, 2*D)
                state = self.rw(state) # (B, 1, D)
                state = self.weight_norm(state)
                h = torch.cat((h, state), dim = 1) # ...--> (B, max_num_steps, D)

        h = self.dropout(h)
        output = torch.bmm(h, paragraphs.transpose(1, 2)) # (B, max_num_steps, N+1)
        output = output + self.bias

        loss = F.binary_cross_entropy_with_logits(output, target, weight = output_mask, reduction = 'mean')
        return loss

    def beam_search(self, input_ids, token_type_ids, attention_mask, examples, tokenizer, retriever, split_chunk):
        beam = self.graph_retriever_config.beam
        B = input_ids.size(0)
        N = self.graph_retriever_config.max_para_num
        
        pred = []
        prob = []

        topk_pred = []
        topk_prob = []
        
        eos_index = N

        init_paragraphs, state = self.encode(input_ids, token_type_ids, attention_mask, split_chunk = split_chunk)

        # Output matrix to be populated
        ps = torch.FloatTensor(N+1, self.s.size(0)).zero_().to(self.s.device) # (N+1, D)
        
        for i in range(B):
            init_context_len = len(examples[i].context)

            # Populating the output matrix by the initial encoding
            ps[:init_context_len, :].copy_(init_paragraphs[i, :init_context_len, :])
            ps[-1, :].copy_(init_paragraphs[i, -1, :])
            encoded_titles = set(examples[i].title_order)
            
            pred_ = [[[], [], 1.0] for _ in range(beam)] # [hist_1, topk_1, score_1], [hist_2, topk_2, score_2], ...
            prob_ = [[] for _ in range(beam)]

            state_ = state[i:i+1] # (1, 1, D)
            state_ = state_.expand(beam, 1, state_.size(2)) # -> (beam, 1, D)
            state_tmp = torch.FloatTensor(state_.size()).zero_().to(state_.device)

            for j in range(self.graph_retriever_config.max_select_num):
                if j > 0:
                    input = [p[0][-1] for p in pred_]
                    input = torch.LongTensor(input).to(ps.device)
                    input = ps[input].unsqueeze(1) # (beam, 1, D)
                    state_ = torch.cat((state_, input), dim = 2) # (beam, 1, 2*D)
                    state_ = self.rw(state_) # (beam, 1, D)
                    state_ = self.weight_norm(state_)

                # Opening new links from the previous predictions (pupulating the output matrix dynamically)
                if j > 0:
                    prev_title_size = len(examples[i].title_order)
                    new_titles = []
                    for b in range(beam):
                        prev_pred = pred_[b][0][-1]

                        if prev_pred == eos_index:
                            continue

                        prev_title = examples[i].title_order[prev_pred]

                        if prev_title not in examples[i].all_linked_paras_dic:

                            if retriever is None:
                                continue
                            else:
                                linked_paras_dic = retriever.get_hyperlinked_abstract_paragraphs(
                                    prev_title, examples[i].question)
                                examples[i].all_linked_paras_dic[prev_title] = {}
                                examples[i].all_linked_paras_dic[prev_title].update(linked_paras_dic)
                                examples[i].all_paras.update(linked_paras_dic)
                        
                        for linked_title in examples[i].all_linked_paras_dic[prev_title]:
                            if linked_title in encoded_titles or len(examples[i].title_order) == N:
                                continue

                            encoded_titles.add(linked_title)
                            new_titles.append(linked_title)
                            examples[i].title_order.append(linked_title)

                    if len(new_titles) > 0:

                        tokens_q = tokenize_question(examples[i].question, tokenizer)
                        input_ids = []
                        input_masks = []
                        segment_ids = []
                        for linked_title in new_titles:
                            linked_para = examples[i].all_paras[linked_title]

                            input_ids_, input_masks_, segment_ids_ = tokenize_paragraph(linked_para, tokens_q, self.graph_retriever_config.max_seq_length, tokenizer)
                            input_ids.append(input_ids_)
                            input_masks.append(input_masks_)
                            segment_ids.append(segment_ids_)

                        input_ids = torch.LongTensor([input_ids]).to(ps.device)
                        token_type_ids = torch.LongTensor([segment_ids]).to(ps.device)
                        attention_mask = torch.LongTensor([input_masks]).to(ps.device)
                        
                        paragraphs, _ = self.encode(input_ids, token_type_ids, attention_mask, split_chunk = split_chunk)
                        paragraphs = paragraphs.squeeze(0)
                        ps[prev_title_size:prev_title_size+len(new_titles)].copy_(paragraphs[:len(new_titles), :])

                        if retriever is not None and self.graph_retriever_config.expand_links:
                            expand_links(examples[i].all_paras, examples[i].all_linked_paras_dic, examples[i].all_paras)
                        
                output = torch.bmm(state_, ps.unsqueeze(0).expand(beam, ps.size(0), ps.size(1)).transpose(1, 2)) # (beam, 1, N+1)
                output = output + self.bias
                output = torch.sigmoid(output)

                output = output.to(self.cpu)
                
                if j == 0:
                    output[:, :, len(examples[i].context):] = 0.0
                else:
                    if len(examples[i].title_order) < N:
                        output[:, :, len(examples[i].title_order):N] = 0.0
                    for b in range(beam):

                        # Omitting previous predictions
                        for k in range(len(pred_[b][0])):
                            output[b, :, pred_[b][0][k]] = 0.0

                        # Links & topK-based pruning
                        if self.graph_retriever_config.pruning_by_links:
                            if pred_[b][0][-1] == eos_index:
                                output[b, :, :eos_index] = 0.0
                                output[b, :, eos_index] = 1.0
                                
                            elif examples[i].title_order[pred_[b][0][-1]] not in examples[i].all_linked_paras_dic:
                                for k in range(len(examples[i].title_order)):
                                    if k not in pred_[b][1]:
                                        output[b, :, k] = 0.0
                                        
                            else:
                                for k in range(len(examples[i].title_order)):
                                    if k not in pred_[b][1] and examples[i].title_order[k] not in examples[i].all_linked_paras_dic[examples[i].title_order[pred_[b][0][-1]]]:
                                        output[b, :, k] = 0.0

                # always >= M before EOS
                if j <= self.graph_retriever_config.min_select_num-1:
                    output[:, :, -1] = 0.0


                score = [p[2] for p in pred_]
                score = torch.FloatTensor(score)
                score = score.unsqueeze(1).unsqueeze(2) # (beam, 1, 1)
                score = output * score
                    
                output = output.squeeze(1) # (beam, N+1)
                score = score.squeeze(1) # (beam, N+1)
                new_pred_ = []
                new_prob_ = []

                b = 0
                while b < beam:
                    s, p = torch.max(score.view(score.size(0)*score.size(1)), dim = 0)
                    s = s.item()
                    p = p.item()
                    row = p // score.size(1)
                    col = p %  score.size(1)

                    if j == 0:
                        score[:, col] = 0.0
                    else:
                        score[row, col] = 0.0

                    p = [[index for index in pred_[row][0]] + [col],
                         output[row].topk(k = 2, dim = 0)[1].tolist(),
                         s]
                    new_pred_.append(p)

                    p = [[p_ for p_ in prb] for prb in prob_[row]] + [output[row].tolist()]
                    new_prob_.append(p)
                    
                    state_tmp[b].copy_(state_[row])
                    b += 1

                pred_ = new_pred_
                prob_ = new_prob_
                state_ = state_.clone()
                state_.copy_(state_tmp)

                if pred_[0][0][-1] == eos_index:
                    break

            topk_pred.append([])
            topk_prob.append([])
            for index__ in range(beam):

                pred_tmp = []
                for index in pred_[index__][0]:
                    if index == eos_index:
                        break
                    pred_tmp.append(index)

                if index__ == 0:
                    pred.append(pred_tmp)
                    prob.append(prob_[0])

                topk_pred[-1].append(pred_tmp)
                topk_prob[-1].append(prob_[index__])

        return pred, prob, topk_pred, topk_prob
