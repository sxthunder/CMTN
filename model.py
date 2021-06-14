import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertLayer
from torch.nn import CrossEntropyLoss

# siamese + raw name 分类
class SiameseClassificationModel(nn.Module):
    def __init__(self, bert_model, bert_config, args):
        super(SiameseClassificationModel, self).__init__()
        self.bert_model = bert_model
        self.hidden_size = bert_config.hidden_size
        self.args = args
        self.classification_loss_fun = CrossEntropyLoss()

        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(self.hidden_size, args.label_nums)

    def get_bert_output(self, input_ids, attention_mask):
        seq_output, cls_output = self.bert_model(input_ids, attention_mask)[:2]
        cls_logits = self.linear(cls_output)
        seq_pool_output = self.seq_pool(seq_output, attention_mask)
        return cls_logits, seq_pool_output

    def seq_pool(self, seq_output, attention_mask):
        #bs,seq_len,hidden bs,seq_len
        if self.args.pool == 'avg':
            seq_pool_output = torch.sum((seq_output * torch.unsqueeze(attention_mask, dim=-1)), dim=1) / torch.sum(attention_mask, dim=-1, keepdim=True)
        else:
            pass
        return seq_pool_output

    def triple_loss_fun(self, pos, neg):
        return F.relu(pos + self.args.margin - neg).mean()

    def get_distance(self,e1,e2):
        if self.args.distance == 'cosine':
            return 1 - F.cosine_similarity(e1,e2)
        elif self.args.distance == 'eu':
            return F.pairwise_distance(e1, e2, p=2)

    def forward(self, batch):
        raw_input_ids, raw_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, raw_name_pos_len = batch

        #bs / neg_num, hidden_size
        raw_cls, raw_output = self.get_bert_output(raw_input_ids, raw_attention_mask)
        _, pos_output = self.get_bert_output(pos_input_ids, pos_attention_mask)

        #bs, hidden
        _, neg_output = self.get_bert_output(neg_input_ids, neg_attention_mask)

        #bs / neg_num, 1
        pos_distance = self.get_distance(raw_output, pos_output)
        neg_distance = self.get_distance(raw_output.repeat((1, self.args.neg_num)).reshape(-1, self.hidden_size), neg_output)

        pos_distance = pos_distance.repeat(1, self.args.neg_num).reshape(-1, 1)

        triple_loss = self.triple_loss_fun(pos_distance, neg_distance)

        raw_name_pos_len_pred = torch.argmax(raw_cls, dim=-1)
        classification_loss = self.classification_loss_fun(raw_cls.squeeze(dim=-1), raw_name_pos_len.squeeze(dim=-1))
        classification_acc = sum(raw_name_pos_len_pred == raw_name_pos_len.squeeze(dim=-1)) / len(raw_cls)

        return triple_loss, classification_loss, classification_acc
        
class BertforClassification(nn.Module):
    def __init__(self, bert, config, args):
        super(BertforClassification, self).__init__()
        self.bert = bert
        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(config.hidden_size, 2)

    def forward(self, batch):
        input_ids, attention_mask, token_type_ids, label = batch
        cls_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        logits = self.linear(self.dropout(cls_output))
        return logits

class BertKeywordsClassification(nn.Module):
    def __init__(self, bert, config, args):
        super(BertKeywordsClassification, self).__init__()
        self.bert = bert
        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(args.dropout)
        # self.seq = nn.Sequential(
        #     nn.Dropout(args.dropout),
        #     nn.Linear(3 * config.hidden_size, config.hidden_size),
        #     nn.Dropout(args.dropout),
        #     nn.Linear(config.hidden_size, 2)
        # )
        self.seq = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(config.hidden_size, 2)
        )

    def forward(self, batch):
        input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, label = batch
        #batch_size, seq_len, hidden_size
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, \
                        token_type_ids=token_type_ids, position_ids=position_ids)[0]

        batch_size = bert_output.size()[0]

        #取前三个token，concat起来, batch_size, 3, 768
        # cls_concat = bert_output[:, :3, :].reshape(batch_size, -1)
        cls_logits = bert_output[:, :3, :].mean(dim=1)
        # print(cls_concat)

        #把关键词都加起来
        # raw_name_att batch_size, seq_len
        # raw_keywords_output = torch.sum((bert_output * torch.unsqueeze(raw_name_att, dim=-1)), dim=1) / (torch.sum(raw_name_att, dim=-1, keepdim=True) + 1e-9) 
        # cand_keywords_output = torch.sum((bert_output * torch.unsqueeze(cand_name_att, dim=-1)), dim=1) / (torch.sum(cand_name_att, dim=-1, keepdim=True) + 1e-9)
        # print(raw_keywords_output)
        # print(cand_keywords_output)

        #将cls和raw以及cand的差值concat起来
        # final_concat = torch.cat([cls_concat, raw_keywords_output - cand_keywords_output, cand_keywords_output - raw_keywords_output], dim=-1)
        # print(final_concat.size())
        # print(final_concat)

        logits = self.seq(cls_logits)
        # print(logits)
        # input()
        return logits

# 仿照keywords attentive的模型，保留body cls和ot cls之外，在bert的最后一层加了一个keywords attentive layer
class BertAttentiveKeywordsClassification(nn.Module):
    def __init__(self, bert, config, args):
        super(BertAttentiveKeywordsClassification, self).__init__()
        self.bert = bert
        self.hidden_size = config.hidden_size
        self.transformer = BertLayer(config)

        self.dropout = nn.Dropout(args.dropout)
        self.seq = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(6 * config.hidden_size, config.hidden_size),
            nn.Dropout(args.dropout),
            nn.Linear(config.hidden_size, 2)
        )

    def forward(self, batch):
        input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, attentive_attention, label = batch
        #batch_size, seq_len, hidden_size
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, \
                        token_type_ids=token_type_ids, position_ids=position_ids)[0]

        batch_size = bert_output.size()[0]

        #取前三个token，concat起来, batch_size, 3, 768
        cls_concat = bert_output[:, :2, :].reshape(batch_size, -1)
        # avg_cls = bert_output

        #bert最后一层继续过一个transformer, 进bert layer之前把attention增加一唯（head维）
        attentive_attention = attentive_attention[:, None, :, :]
        attentive_output = self.transformer(hidden_states=bert_output, attention_mask=attentive_attention)[0]

        #把关键词都加起来
        # raw_name_att batch_size, seq_len
        raw_keywords_output = torch.sum((attentive_output * torch.unsqueeze(raw_name_att, dim=-1)), dim=1) / (torch.sum(raw_name_att, dim=-1, keepdim=True) + 1e-9) 
        cand_keywords_output = torch.sum((attentive_output * torch.unsqueeze(cand_name_att, dim=-1)), dim=1) / (torch.sum(cand_name_att, dim=-1, keepdim=True) + 1e-9)
        # print(raw_keywords_output)
        # print(cand_keywords_output)

        #将cls和raw以及cand的差值concat起来
        final_concat = torch.cat([cls_concat, raw_keywords_output, cand_keywords_output, raw_keywords_output - cand_keywords_output, cand_keywords_output - raw_keywords_output], dim=-1)
        # print(final_concat.size())
        # print(final_concat)

        logits = self.seq(final_concat)
        # print(logits)
        # input()
        return logits

# cls做二分类，加上关键词做分类，融合二者结果做输出
class BertCodingClassification(nn.Module):
    def __init__(self, bert, config, args):
        super(BertCodingClassification, self).__init__()
        self.bert = bert
        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(args.dropout)
        self.cls_linear = nn.Linear(config.hidden_size, 2)
        self.keywords_linear = nn.Linear(config.hidden_size, 2)

    def get_cls_and_keywords_output(self, input_ids, attention_mask, token_type_ids, position_ids):
        seq_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids)[0]
        cls_output = self.cls_linear(self.dropout(seq_output[:, 0, :]))
        keywords_output = self.keywords_linear(self.dropout(seq_output[:,1,:])) 
        return cls_output, keywords_output


    def forward(self, batch):
        input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, code_type_label, label = batch
        # input_ids, attention_mask, token_type_ids, label = batch
        cls_logits, keywords_logits = self.get_cls_and_keywords_output(input_ids, attention_mask, token_type_ids, position_ids)
        return cls_logits, keywords_logits

# bert直接做多分类
class BertMultiClassification(nn.Module):
    def __init__(self, bert, config, num_labels, args):
        super(BertMultiClassification, self).__init__()
        self.bert = bert
        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(args.dropout)
        self.cls_linear = nn.Linear(config.hidden_size, num_labels)

    def forward(self, batch):
        input_ids, attention_mask = batch
        cls_output = self.bert(input_ids, attention_mask)[1]
        cls_logits = self.cls_linear(self.dropout(cls_output))
        return cls_logits
