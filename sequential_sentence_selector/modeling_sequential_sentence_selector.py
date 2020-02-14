from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter

class BertForSequentialSentenceSelector(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForSequentialSentenceSelector, self).__init__(config)

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

    def encode(self, input_ids, token_type_ids, attention_mask, split_chunk = None):
        B = input_ids.size(0)
        N = input_ids.size(1)
        input_ids = input_ids.contiguous().view(input_ids.size(0)*input_ids.size(1), input_ids.size(2))
        token_type_ids = token_type_ids.contiguous().view(token_type_ids.size(0)*token_type_ids.size(1), token_type_ids.size(2))
        attention_mask = attention_mask.contiguous().view(attention_mask.size(0)*attention_mask.size(1), attention_mask.size(2))

        # [CLS] vectors for Q-P pairs
        if split_chunk is None:
            encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
            pooled_output = encoded_layers[:, 0]
        else:
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
                    
        paragraphs = pooled_output.view(pooled_output.size(0)//N, N, pooled_output.size(1)) # (B, N, D)
        EOE = self.eos.unsqueeze(0).unsqueeze(0) # (1, 1, D)
        EOE = EOE.expand(paragraphs.size(0), EOE.size(1), EOE.size(2)) # (B, 1, D)
        EOE = self.bert.encoder.layer[-1].output.LayerNorm(EOE)
        paragraphs = torch.cat((paragraphs, EOE), dim = 1) # (B, N+1, D)

        # Initial state
        state = self.s.expand(paragraphs.size(0), 1, self.s.size(0))
        state = self.weight_norm(state)

        return paragraphs, state
        
    def forward(self, input_ids, token_type_ids, attention_mask, output_mask, target, target_ids, max_sf_num):

        paragraphs, state = self.encode(input_ids, token_type_ids, attention_mask)

        for i in range(max_sf_num+1):
            if i == 0:
                h = state
            else:
                for j in range(target_ids.size(0)):
                    index = target_ids[j, i-1]
                    input_ = paragraphs[j:j+1, index:index+1, :] # (B, 1, D)

                    if j == 0:
                        input = input_
                    else:
                        input = torch.cat((input, input_), dim = 0)

                state = torch.cat((state, input), dim = 2) # (B, 1, 2*D)
                state = self.rw(state) # (B, 1, D)
                state = self.weight_norm(state)
                h = torch.cat((h, state), dim = 1) # ...--> (B, max_num_steps, D)

        h = self.dropout(h)
        output = torch.bmm(h, paragraphs.transpose(1, 2)) # (B, max_num_steps, N+1)
        output = output + self.bias
        loss = F.binary_cross_entropy_with_logits(output, target, weight = output_mask, reduction = 'mean')
        return loss

    def beam_search(self, input_ids, token_type_ids, attention_mask, output_mask, max_num_steps, examples, beam = 2):

        B = input_ids.size(0)
        paragraphs, state = self.encode(input_ids, token_type_ids, attention_mask, split_chunk = 300)
        
        pred = []
        prob = []

        topk_pred = []
        topk_prob = []
        
        eoe_index = paragraphs.size(1)-1

        output_mask = output_mask.to(self.cpu)
        
        for i in range(B):
            pred_ = [[[], 1.0, 0] for _ in range(beam)] # [hist_1, score_1, len_1], [hist_2, score_2, len_2], ...
            prob_ = [[] for _ in range(beam)]

            state_ = state[i:i+1] # (1, 1, D)
            state_ = state_.expand(beam, 1, state_.size(2)) # -> (beam, 1, D)
            state_tmp = torch.FloatTensor(state_.size()).zero_().to(state_.device)
            ps = paragraphs[i:i+1] # (1, N+1, D)
            ps = ps.expand(beam, ps.size(1), ps.size(2)) # -> (beam, N+1, D)

            for j in range(max_num_steps):
                if j > 0:
                    input = [p[0][-1] for p in pred_]
                    input = torch.LongTensor(input).to(paragraphs.device)
                    input = ps[0][input].unsqueeze(1) # (beam, 1, D)
                    state_ = torch.cat((state_, input), dim = 2) # (beam, 1, 2*D)
                    state_ = self.rw(state_) # (beam, 1, D)
                    state_ = self.weight_norm(state_)

                output = torch.bmm(state_, ps.transpose(1, 2)) # (beam, 1, N+1)
                output = output + self.bias
                output = torch.sigmoid(output)

                output = output.to(self.cpu)

                if j == 0:
                    output = output * output_mask[i:i+1, 0:1, :]
                else:
                    for b in range(beam):
                        for k in range(len(pred_[b][0])):
                            output[b:b+1] *= output_mask[i:i+1, pred_[b][0][k]+1:pred_[b][0][k]+2, :]

                e = examples[i]
                # Predict at least 1 sentence anyway
                if j <= 0:
                    output[:, :, -1] = 0.0
                # No further constraints for single title
                elif len(e.titles) == 1:
                    pass
                else:
                    for b in range(beam):
                        sfs = set()
                        for p in pred_[b][0]:
                            if p == eoe_index:
                                break
                            offset = 0
                            for k in range(len(e.titles)):
                                if p >= offset and p < offset+len(e.context[e.titles[k]]):
                                    sfs.add(e.titles[k])
                                    break
                                offset += len(e.context[e.titles[k]])
                        if len(sfs) == 1:
                            output[b, :, -1] = 0.0

                score = [p[1] for p in pred_]
                score = torch.FloatTensor(score)
                score = score.unsqueeze(1).unsqueeze(2) # (beam, 1, 1)
                score = output * score
                    
                output = output.squeeze(1) # (beam, N+1)
                score = score.squeeze(1) # (beam, N+1)
                new_pred_ = []
                new_prob_ = []

                for b in range(beam):
                    s, p = torch.max(score.view(score.size(0)*score.size(1)), dim = 0)
                    s = s.item()
                    p = p.item()
                    row = p // score.size(1)
                    col = p %  score.size(1)

                    p = [[index for index in pred_[row][0]] + [col],
                         score[row, col].item(),
                         pred_[row][2] + (1 if col != eoe_index else 0)]
                    new_pred_.append(p)

                    p = [[p_ for p_ in prb] for prb in prob_[row]] + [output[row].tolist()]
                    new_prob_.append(p)
                    
                    state_tmp[b].copy_(state_[row])

                    if j == 0:
                        score[:, col] = 0.0
                    else:
                        score[row, col] = 0.0

                pred_ = new_pred_
                prob_ = new_prob_
                state_ = state_.clone()
                state_.copy_(state_tmp)

                if pred_[0][0][-1] == eoe_index:
                    break

            topk_pred.append([])
            topk_prob.append([])
            for index__ in range(beam):

                pred_tmp = []
                for index in pred_[index__][0]:
                    if index == eoe_index:
                        break
                    pred_tmp.append(index)

                if index__ == 0:
                    pred.append(pred_tmp)
                    prob.append(prob_[0])

                topk_pred[-1].append(pred_tmp)
                topk_prob[-1].append(prob_[index__])

        return pred, prob, topk_pred, topk_prob
