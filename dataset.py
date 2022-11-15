import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler, RandomSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import tokenize, read_dict, load_keywords, padding_keywords_att, get_coding_type

#构建所有的dataset
class TripleDataset(Dataset):
    #args.neg_num:代表负例的个数, 如果neg_list is None, 则根据args.neg_sample进行采样，否则直接使用neg_list做负例（hard sample会提供neg_list）
    def __init__(self, data_list, standard_name_list, code_to_name, name_to_code, tokenizer, args, logger, neg_list=None):
        self.data_list = data_list # train val answer
        self.standard_name_list = standard_name_list #标准名词表
        self.code_to_name = code_to_name
        self.name_to_code = name_to_code
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.neg_num = args.neg_num
        self.neg_sample = args.neg_sample
        self.batch_size = args.train_batch_size
        self.neg_list = neg_list # len=len(data_list), 每个item长度为该raw_name对应standard name 个数的一个list, 而其中又包含了args.neg_num个负例
        self.device = args.device
        self.logger = logger
        self.add_keywords = args.add_keywords

        #如果是tf_idf，提前构造好
        if self.neg_sample == 'tf_idf' or self.neg_sample == 'online':
            self.logger.info('generate tf idf model......')
            self.tf_idf_neg_list = self.get_tf_idf_neg_list()

        if self.neg_sample == 'tree_index':
            self.logger.info('generate tree index......')
            self.tree_index_dict = self.get_tree_index()

        # 采用关键词替换采样或者使用关键词信息时，读取对应的字典及关键词信息
        if self.neg_sample == 'keyword_replace':
            self.dict, self.key_words = load_keywords(logger)

        self.init()

    def get_tree_index(self):
        tree_index_dict = {}
        for k, v in self.code_to_name.items():
            index = k.split('.')[0]
            index_list = tree_index_dict.get(index, [])
            index_list.append(self.standard_name_list.index(v))
            tree_index_dict[index] = index_list
        return tree_index_dict

    def get_tf_idf_neg_list(self):
        tf_idf = TfidfVectorizer(analyzer='char')
        tf_idf.fit(self.standard_name_list)
        raw_name_list = [x[0] for x in self.data_list]
        
        raw_feature = tf_idf.transform(raw_name_list)
        standard_feature = tf_idf.transform(self.standard_name_list)
        self.logger.info('generate tf idf similarity')
        similarity = (raw_feature * standard_feature.T).toarray() #len(raw_name) * len(standard_name)
        similarity = np.argpartition(-similarity, self.neg_num + 10, axis=-1)

        tf_idf_neg_list = []
        #这里由于同一个raw_name的每个pos_name对应的neg_list是一样的
        for sim_score, (_, pos_name_list) in zip(similarity, self.data_list):
            neg_list = [self.standard_name_list[x] for x in sim_score if self.standard_name_list[x] not in pos_name_list][:self.neg_num]
            assert len(neg_list) == self.neg_num
            tf_idf_neg_list.append([neg_list for _ in range(len(pos_name_list))])
            assert len(tf_idf_neg_list[-1]) == len(pos_name_list)
        
        assert len(tf_idf_neg_list) == len(self.data_list)
        return tf_idf_neg_list

    def __len__(self):
        # return len(self.data_list) * self.neg_num
        return len(self.item_data_list)

    def __getitem__(self, idx):
        raw_name, pos_name, neg_name, raw_name_pos_len = self.item_data_list[idx]
        # 这里的设置是为了使得同样的raw name和pos name不会重复进入到bert进行计算,要求batch_size = n * neg_num,
        # 每个batch中 raw name 和pos name的个数为 batch_size // neg_num, neg name的个数为batch_size
        if idx % self.neg_num != 0:
            raw_name_pos_len = -1
            raw_input_ids = [0] * self.max_len
            raw_attention_mask = [0] * self.max_len
            pos_input_ids = [0] * self.max_len
            pos_attention_mask = [0] * self.max_len
        else:
            raw_input_ids, raw_attention_mask = tokenize(raw_name, self.tokenizer, self.max_len)
            pos_input_ids, pos_attention_mask = tokenize(pos_name, self.tokenizer, self.max_len)

        neg_input_ids, neg_attention_mask = tokenize(neg_name, self.tokenizer, self.max_len)

        return  (raw_input_ids, raw_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, raw_name_pos_len)

    def remove_zero_line(self, m, v=0):
        return m[~torch.all(m == v, dim=-1)]

    def collate_fn(self, batch):
        batch_item_len = len(batch[0])
        batch_output = [torch.tensor([item[i] for item in batch]).to(self.device) for i in range(batch_item_len)]

        raw_input_ids, raw_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, raw_name_pos_len = batch_output

        raw_input_ids = self.remove_zero_line(raw_input_ids)
        raw_attention_mask = self.remove_zero_line(raw_attention_mask)
        pos_input_ids = self.remove_zero_line(pos_input_ids)
        pos_attention_mask = self.remove_zero_line(pos_attention_mask)
        raw_name_pos_len = self.remove_zero_line(raw_name_pos_len.unsqueeze(dim=-1), v=-1)
        
        return (raw_input_ids, raw_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask, raw_name_pos_len)

    def init(self):
        self.item_data_list = [] # 处理后的data_list
        for idx in range(len(self.data_list)):
            raw_name, pos_name_list = self.data_list[idx]
            neg_name_list = self.sample_neg_name_list(raw_name, pos_name_list, idx)
            for pos_name, neg_names in zip(pos_name_list, neg_name_list):
                pos_len = len(pos_name_list) - 1 if len(pos_name_list) < 3 else 2
                self.item_data_list += [(raw_name, pos_name, neg_name, pos_len) for neg_name in neg_names]
        
    def sample_neg_name_list(self, raw_name, pos_name_list, idx):
        #相当于oneline
        if self.neg_list is not None:
            # self.logger.info('{}, {}'.format(self.data_list[idx], self.neg_list[idx]))
            return self.neg_list[idx]

        #完全随机, 如果是online，第一波也会随机
        elif self.neg_sample == 'random' or self.neg_sample == 'online':
            return [np.random.choice(self.standard_name_list, self.neg_num, replace=False).tolist() for _ in range(len(pos_name_list))]
        
        #根据raw_name计算和standard_name_list的tf_idf
        elif self.neg_sample == 'tf_idf':
            return self.tf_idf_neg_list[idx]

        #根据pos_name_list找同一个class下的词
        elif self.neg_sample == 'tree_index':
            item_neg_list = []
            for pos_name in pos_name_list:
                pos_name_index = self.name_to_code[pos_name].split('.')[0]
                index_name_list = self.tree_index_dict[pos_name_index]
                neg_list = []
                while len(neg_list) < self.neg_num:
                    neg_index = np.random.choice(index_name_list)
                    neg_name = self.standard_name_list[neg_index]
                    if neg_name not in pos_name_list:
                        neg_list.append(neg_name)
                assert len(neg_list) == self.neg_num
                item_neg_list.append(neg_list)
            assert len(item_neg_list) == len(pos_name_list)
            return item_neg_list

        elif self.neg_sample == 'keyword_replace':
            item_neg_list = []
            for pos_name in pos_name_list:
                # 先从pos name和raw name里随机抽取一个，然后随机抽取部位、入路、术式进行替换
                temp_list = [pos_name, raw_name]
                temp_neg_list = []
                while len(temp_neg_list) < self.neg_num:
                    neg_name = np.random.choice(temp_list)
                    assert neg_name in self.key_words
                    neg_name = self.replace_keyword(neg_name)
                    self.generate_term_keywords(neg_name)
                    temp_neg_list.append(neg_name)
                item_neg_list.append(temp_neg_list)
            return item_neg_list
        else:
            raise Exception('neg sample 或neg list配置错误')

    #对于替换关键词生成的负例，通过此方法构造其 对应关键词的attention，并统一存储在self.keywords里面
    def generate_term_keywords(self, name):
        for t in ['body', 'rulu', 'ot']:
            att = [0] * len(name)
            self.key_words[name][t] = []
            for keyword in self.key_words[name][t]:
                if keyword not in name:
                    keyword = keyword[:-1]
                
                start = name.index(keyword)
                end = start + len(keyword)
                att[start:end] = [1] * len(keyword)
            self.key_words[name]['{}_att'.format(t)] = att

    def replace_keyword(self, name):
        neg_name = name.copy()
        temp_d = {}
        for t in ['body', 'rulu', 'ot']:
            name_dict_list = self.key_words[name][t]
            dic_list = self.dict[t]
            if len(name_dict_list) == 0:
                continue
            old_term = np.random.choice(name_dict_list)
            new_term = np.random.choice(dic_list)
            if old_term not in name:
                old_term = old_term[:-1]
            start = name.index(old_term)
            end = start + len(old_term)
            neg_name = neg_name[:start] + new_term + neg_name[end:]
            neg_name_dict_list = name_dict_list.copy()
            try:
                neg_name_dict_list.remove(old_term)
            except:
                neg_name_dict_list.remove(old_term + '术')
            neg_name_dict_list.append(new_term)
            temp_d[t] = neg_name_dict_list
        self.key_words[neg_name] = temp_d
        return neg_name

#构建val和test以及icd的dataset
class TermDataset(Dataset):
    def __init__(self, data_list, tokenizer, args, is_icd, logger=None):
        self.data_list = data_list
        self.max_len = args.max_len
        self.batch_size = args.val_batch_size
        self.device = args.device
        self.tokenizer = tokenizer
        self.args = args
        self.is_icd = is_icd
        self.logger = logger
        

    def __getitem__(self, idx):
        if self.is_icd:
            standard_name = self.data_list[idx]
            input_ids, attention_mask = tokenize(standard_name, self.tokenizer, self.max_len)
            return_tuple = (input_ids, attention_mask) 
        else:
            raw_name, pos_name_list = self.data_list[idx]
            input_ids, attention_mask = tokenize(raw_name, self.tokenizer, self.max_len)
            pos_len = len(pos_name_list) - 1 if len(pos_name_list) < 3 else 2

            return_tuple = (input_ids, attention_mask, pos_name_list, raw_name, pos_len)
        
        return return_tuple
        

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        batch_item_len = len(batch[0])

        if self.is_icd:
            input_ids, attention_mask = \
                [torch.tensor([item[i] for item in batch]).to(self.device) for i in range(batch_item_len)]
            return input_ids, attention_mask
                
        else:
            input_ids, attention_mask, pos_len = \
                [torch.tensor([item[i] for item in batch]).to(self.device) for i in range(batch_item_len) if i not in [2, 3]]
            batch_pos_name_list = [item[2] for item in batch]
            batch_raw_name_list = [item[3] for item in batch]
            return input_ids, attention_mask, batch_pos_name_list, batch_raw_name_list, pos_len
        
# 带有keywords的rerank的dataset
class RerankKeywordDataset(Dataset):
    def __init__(self, data_list, tokenizer, args, logger):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.device = args.device
        self.logger = logger
        self.cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.body_cls = self.tokenizer.convert_tokens_to_ids('[unused1]')
        self.ot_cls = self.tokenizer.convert_tokens_to_ids('[unused2]')
        self.args = args
        self.init()

        self.d, self.key_words = load_keywords(self.logger)

    def init(self):
        self.item_data_list = []
        self.raw_name_num_pred = {}
        self.raw_name_num_label = {}
        self.raw_name_pos_names = {}
        for item in self.data_list:
            raw_name, pos_names, pos_names_score, cand_names, cand_names_score, num_pred, num_label = item
            for x, y in zip(cand_names, cand_names_score):
                label = 1 if x in pos_names else 0
                self.item_data_list.append((raw_name, x, label, y))
            self.raw_name_num_label[raw_name] = num_label
            self.raw_name_num_pred[raw_name] = num_pred
            self.raw_name_pos_names[raw_name] = pos_names

    def __len__(self):
        return len(self.item_data_list)

    def tokenize(self, raw_name, cand_name, raw_name_body_att, raw_name_ot_att, cand_name_body_att, cand_name_ot_att):
        raw_name_input_ids = self.tokenizer.encode(list(raw_name), add_special_tokens=False)
        cand_name_input_ids = self.tokenizer.encode(list(cand_name), add_special_tokens=False)
        assert len(raw_name_input_ids) == len(raw_name)
        assert len(cand_name_input_ids) == len(cand_name)

        #encode
        input_ids = [self.cls] + [self.body_cls] + [self.ot_cls] + raw_name_input_ids + [self.sep] + cand_name_input_ids + [self.sep]
        
        #token_type_ids
        token_type_ids = [0] * (len(raw_name_input_ids) + 4)
        token_type_ids += [1] * (len(input_ids) - len(token_type_ids))

        #position embedding: cls body ot 都设置为0
        if self.args.cls_position == 'zero':
            position_ids = [0, 0, 0] + [i + 1 for i in range(len(input_ids) - 3)]
        else:
            position_ids = [i for i in range(len(input_ids))]

        #分别标齐raw words和cand name的关键词位置
        raw_name_att = [0] * 3 + [1 if (raw_name_body_att[i] == 1 or raw_name_ot_att[i] == 1) else 0 for i in range(len(raw_name_body_att))] \
                + [0] * (self.max_len - len(raw_name_body_att) - 3)
        cand_name_att = [0] * (4 + len(raw_name_body_att)) + [1 if (cand_name_body_att[i] == 1 or cand_name_ot_att[i] == 1) else 0 for i in range(len(cand_name_body_att))]
        cand_name_att = cand_name_att[:self.max_len]
        cand_name_att += [0] * (self.max_len - len(cand_name_att))
        assert len(raw_name_att) == self.max_len, print(len(raw_name_att), len(raw_name_body_att))
        assert len(cand_name_att) == self.max_len

        #得到body和ot在encode后的位置信息，方便构造attention矩阵
        body_att_index = [i + 3 for i in range(len(raw_name_body_att)) if raw_name_body_att[i] == 1] + \
                            [i + 4 + len(raw_name_input_ids) for i in range(len(cand_name_body_att)) if cand_name_body_att[i] == 1]
        ot_att_index = [i + 3 for i in range(len(raw_name_ot_att)) if raw_name_ot_att[i] == 1] + \
                            [i + 4 + len(raw_name_input_ids) for i in range(len(cand_name_ot_att)) if cand_name_ot_att[i] == 1]

        attention_mask = [[1] + [0, 0] + [1] * (len(input_ids) - 3) for _ in range(len(input_ids))]
        #body_cls只能看见所有的body词
        attention_mask[1] = [0, 1, 0] + [1 if i in body_att_index else 0 for i in range(3, len(input_ids))]
        attention_mask[2] = [0, 0, 1] + [1 if i in ot_att_index else 0 for i in range(3, len(input_ids))]

        #手术词能够看到body cls, 同时对应的keywords attention也为1
        for i in body_att_index:
            # if i < self.max_len:
            attention_mask[i][1] = 1
        
        #术式词能够看到ot clf,
        for i in ot_att_index:
            # if i < self.max_len:
            attention_mask[i][2] = 1
        
        #进行padding
        for i in range(len(attention_mask)):
            attention_mask[i] += [0] * (self.max_len - len(attention_mask[i])) 
        
        attention_mask += [[0] * self.max_len] * (self.max_len - len(attention_mask))
        # print(attention_mask)
        assert len(attention_mask) == self.max_len
        input_ids += [0] * (self.max_len - len(input_ids))
        position_ids += [0] * (self.max_len - len(position_ids))
        token_type_ids += [0] * (self.max_len - len(token_type_ids))
        
        # input()
        # print(input_ids)
        # print(position_ids)
        # print(token_type_ids)

        # print(raw_name)
        # print(raw_name_body_att)
        # print(raw_name_ot_att)
        # print(raw_name_input_ids)
        # print(raw_name_att)

        # print(cand_name)
        # print(cand_name_body_att)
        # print(cand_name_ot_att)
        # print(cand_name_input_ids)
        # print('1')
        # print(cand_name_att)
        # print('222')

        # for z in attention_mask:
        #     print(z)

        # input()


        return input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att


    def __getitem__(self, idx):
        raw_name, cand_name, label, cand_score = self.item_data_list[idx]
        #得到各自的att
        raw_name_body_att = self.key_words[raw_name]['body_att']
        raw_name_ot_att = self.key_words[raw_name]['ot_att']
        cand_name_body_att = self.key_words[cand_name]['body_att']
        cand_name_ot_att = self.key_words[cand_name]['ot_att']
        assert len(raw_name) == len(raw_name_body_att)
        assert len(cand_name) == len(cand_name_ot_att)
        # 5是因为cls + 2个sep + body_cls + ot_cls
        if len(raw_name) + len(cand_name) + 5 > self.max_len:
            if len(raw_name) > len(cand_name):
                end_index = self.max_len - len(cand_name) - 5
                raw_name = raw_name[:end_index]  
                raw_name_body_att = raw_name_body_att[:end_index]
                raw_name_ot_att = raw_name_ot_att[:end_index]
            else:
                end_index = self.max_len - len(raw_name) - 5
                cand_name = cand_name[:end_index]
                cand_name_body_att = cand_name_body_att[:end_index]
                cand_name_ot_att = cand_name_ot_att[:end_index]
        input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att = self.tokenize(raw_name, cand_name, raw_name_body_att, raw_name_ot_att, cand_name_body_att, cand_name_ot_att)

        return input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, label, cand_score

    def collate_fn(self, batch):
        batch_item_len = len(batch[0])
        input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, label, cand_score = \
            [torch.tensor([item[x] for item in batch]).to(self.device) for x in range(batch_item_len)]


        return input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, label, cand_score
    
# bert二分类rerank的模型
class RerankDataset(Dataset):
    def __init__(self, data_list, tokenizer, args):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.device = args.device
        self.init()

    def init(self):
        self.item_data_list = []
        self.raw_name_num_pred = {}
        self.raw_name_num_label = {}
        self.raw_name_pos_names = {}
        for item in self.data_list:
            raw_name, pos_names, pos_names_score, cand_names, cand_names_score, num_pred, num_label = item
            for x, y in zip(cand_names, cand_names_score):
                label = 1 if x in pos_names else 0
                self.item_data_list.append((raw_name, x, label, y))
            self.raw_name_num_label[raw_name] = num_label
            self.raw_name_num_pred[raw_name] = num_pred
            self.raw_name_pos_names[raw_name] = pos_names

    def __len__(self):
        return len(self.item_data_list)

    def __getitem__(self, idx):
        raw_name, cand_name, label, cand_score = self.item_data_list[idx]
        if len(raw_name) + len(cand_name) + 3 > self.max_len:
            if len(raw_name) > len(cand_name):
                raw_name = raw_name[:self.max_len - len(cand_name) - 3]  
            else:
                cand_name = cand_name[:self.max_len - len(raw_name) - 3]

        token_result = self.tokenizer.encode_plus(raw_name, cand_name, padding='max_length', max_length=self.max_len)
        input_ids = token_result['input_ids']
        token_type_ids = token_result['token_type_ids']
        attention_mask = token_result['attention_mask']
        return input_ids, attention_mask, token_type_ids, label, cand_score


    def collate_fn(self, batch):
        batch_item_len = len(batch[0])
        input_ids, attention_mask, token_type_ids, label, cand_score = \
            [torch.tensor([item[x] for item in batch]).to(self.device) for x in range(batch_item_len)]

        return input_ids, attention_mask, token_type_ids, label, cand_score

# 带有keywords的attentive模型
class RerankAttentiveDataset(Dataset):
    def __init__(self, data_list, tokenizer, args, logger):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.device = args.device
        self.logger = logger
        self.cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.keyword_cls = self.tokenizer.convert_tokens_to_ids('[unused1]')
        # self.ot_cls = self.tokenizer.convert_tokens_to_ids('[unused2]')
        self.args = args
        self.init()

        self.d, self.key_words = load_keywords(self.logger)

    def init(self):
        self.item_data_list = []
        self.raw_name_num_pred = {}
        self.raw_name_num_label = {}
        self.raw_name_pos_names = {}
        for item in self.data_list:
            raw_name, pos_names, pos_names_score, cand_names, cand_names_score, num_pred, num_label = item
            for x, y in zip(cand_names, cand_names_score):
                label = 1 if x in pos_names else 0
                self.item_data_list.append((raw_name, x, label, y))
            self.raw_name_num_label[raw_name] = num_label
            self.raw_name_num_pred[raw_name] = num_pred
            self.raw_name_pos_names[raw_name] = pos_names

    def __len__(self):
        return len(self.item_data_list)

    def tokenize(self, raw_name, cand_name, raw_name_body_att, raw_name_ot_att, cand_name_body_att, cand_name_ot_att):
        raw_name_input_ids = self.tokenizer.encode(list(raw_name), add_special_tokens=False)
        cand_name_input_ids = self.tokenizer.encode(list(cand_name), add_special_tokens=False)
        assert len(raw_name_input_ids) == len(raw_name)
        assert len(cand_name_input_ids) == len(cand_name)

        #encode
        input_ids = [self.cls]  + [self.keyword_cls] + raw_name_input_ids + [self.sep] + cand_name_input_ids + [self.sep]
        
        #token_type_ids
        token_type_ids = [0] * (len(raw_name_input_ids) + 4)
        token_type_ids += [1] * (len(input_ids) - len(token_type_ids))

        #position embedding: cls body ot 都设置为0
        if self.args.cls_position == 'zero':
            position_ids = [0, 0] + [i + 1 for i in range(len(input_ids) - 2)]
        else:
            position_ids = [i for i in range(len(input_ids))]

        #分别标齐raw words和cand name的关键词位置
        raw_name_att = [0] * 2 + [1 if (raw_name_body_att[i] == 1 or raw_name_ot_att[i] == 1) else 0 for i in range(len(raw_name_body_att))] \
                + [0] * (self.max_len - len(raw_name_body_att) - 2)
        cand_name_att = [0] * (3 + len(raw_name_body_att)) + [1 if (cand_name_body_att[i] == 1 or cand_name_ot_att[i] == 1) else 0 for i in range(len(cand_name_body_att))]
        cand_name_att = cand_name_att[:self.max_len]
        cand_name_att += [0] * (self.max_len - len(cand_name_att))
        assert len(raw_name_att) == self.max_len, print(len(raw_name_att), len(raw_name_body_att))
        assert len(cand_name_att) == self.max_len

        #*******************生成过bert的attention*********************
        attention_mask = [[1, 0] + [1] * (len(input_ids) - 3) for _ in range(len(input_ids))]
        #区别与上一个datset，这里只放一个多余的token作为keywords_cls，注意所有的keywords
        attention_mask[1] = [0, 1] + raw_name_att[2:len(raw_name_body_att)+2] + [0] + cand_name_att[3+len(raw_name_body_att):3+len(raw_name_body_att)+len(cand_name_body_att)]

        #所有的关键词都可以看到keywords cls
        for i in range(len(raw_name_att)):
            if raw_name_att[i] == 1 or cand_name_att[i] == 1:
                attention_mask[i][1] = 1
        
        #进行padding
        for i in range(len(attention_mask)):
            attention_mask[i] += [0] * (self.max_len - len(attention_mask[i])) 
        
        attention_mask += [[0] * self.max_len] * (self.max_len - len(attention_mask))

        #*************生成key attentive的attention****************
        attentive_attention = [[0] * len(input_ids) for _ in range(len(input_ids))]

        #raw能看到cand的所有关键词
        for i in range(3, len(raw_name_input_ids) + 3):
            attentive_attention[i] = cand_name_att
        
        #cand能看到raw的所有关键词
        for i in range(4 + len(raw_name_input_ids), len(input_ids)):
            attentive_attention[i] = raw_name_att

        for i in range(len(attentive_attention)):
            attentive_attention[i] += [0] * (self.max_len - len(attentive_attention[i]))

        attentive_attention += [[0] * self.max_len] * (self.max_len - len(attentive_attention))

        # print(attention_mask)
        assert len(attention_mask) == self.max_len
        input_ids += [0] * (self.max_len - len(input_ids))
        position_ids += [0] * (self.max_len - len(position_ids))
        token_type_ids += [0] * (self.max_len - len(token_type_ids))
        
        # input()
        # print(input_ids)
        # print(position_ids)
        # print(token_type_ids)

        # print(raw_name)
        # print(raw_name_body_att)
        # print(raw_name_ot_att)
        # print(raw_name_input_ids)
        # print(raw_name_att)

        # print(cand_name)
        # print(cand_name_body_att)
        # print(cand_name_ot_att)
        # print(cand_name_input_ids)
        # print(cand_name_att)
        # print('##')
        # # for z in attentive_attention:
        # #     print(z)
        # # print('##')

        # for z in attention_mask:
        #     print(z)

        # input()


        return input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, attentive_attention


    def __getitem__(self, idx):
        raw_name, cand_name, label, cand_score = self.item_data_list[idx]
        #得到各自的att
        raw_name_body_att = self.key_words[raw_name]['body_att']
        raw_name_ot_att = self.key_words[raw_name]['ot_att']
        cand_name_body_att = self.key_words[cand_name]['body_att']
        cand_name_ot_att = self.key_words[cand_name]['ot_att']
        assert len(raw_name) == len(raw_name_body_att)
        assert len(cand_name) == len(cand_name_ot_att)
        # 5是因为cls + 2个sep + body_cls + ot_cls
        if len(raw_name) + len(cand_name) + 5 > self.max_len:
            if len(raw_name) > len(cand_name):
                end_index = self.max_len - len(cand_name) - 5
                raw_name = raw_name[:end_index]  
                raw_name_body_att = raw_name_body_att[:end_index]
                raw_name_ot_att = raw_name_ot_att[:end_index]
            else:
                end_index = self.max_len - len(raw_name) - 5
                cand_name = cand_name[:end_index]
                cand_name_body_att = cand_name_body_att[:end_index]
                cand_name_ot_att = cand_name_ot_att[:end_index]
        input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, attentive_attention = self.tokenize(raw_name, cand_name, raw_name_body_att, raw_name_ot_att, cand_name_body_att, cand_name_ot_att)

        return input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, attentive_attention, label, cand_score

    def collate_fn(self, batch):
        batch_item_len = len(batch[0])
        input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, attentive_attention, label, cand_score = \
            [torch.tensor([item[x] for item in batch]).to(self.device) for x in range(batch_item_len)]


        return input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, attentive_attention, label, cand_score
            
# 带有keywords的rerank的dataset
class RerankWithCodeDataset(Dataset):
    def __init__(self, data_list, tokenizer, args, logger, name_to_code):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.device = args.device
        self.logger = logger
        self.cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.body_cls = self.tokenizer.convert_tokens_to_ids('[unused1]')
        self.args = args
        self.name_to_code = name_to_code
        self.init()

        self.d, self.key_words = load_keywords(self.logger)

        self.code_dict = get_coding_type()

    def init(self):
        self.item_data_list = []
        self.raw_name_num_pred = {}
        self.raw_name_num_label = {}
        self.raw_name_pos_names = {}
        for item in self.data_list:
            raw_name, pos_names, pos_names_score, cand_names, cand_names_score, num_pred, num_label = item
            for x, y in zip(cand_names, cand_names_score):
                label = 1 if x in pos_names else 0
                self.item_data_list.append((raw_name, x, label, y))
            self.raw_name_num_label[raw_name] = num_label
            self.raw_name_num_pred[raw_name] = num_pred
            self.raw_name_pos_names[raw_name] = pos_names

    def __len__(self):
        return len(self.item_data_list)

    def tokenize(self, raw_name, cand_name, raw_name_body_att, cand_name_body_att):
        raw_name_input_ids = self.tokenizer.encode(list(raw_name), add_special_tokens=False)
        cand_name_input_ids = self.tokenizer.encode(list(cand_name), add_special_tokens=False)
        assert len(raw_name_input_ids) == len(raw_name)
        assert len(cand_name_input_ids) == len(cand_name)

        #encode
        input_ids = [self.cls] + [self.body_cls] + raw_name_input_ids + [self.sep] + cand_name_input_ids + [self.sep]
        
        #token_type_ids
        token_type_ids = [0] * (len(raw_name_input_ids) + 3)
        token_type_ids += [1] * (len(input_ids) - len(token_type_ids))

        #position embedding: cls body ot 都设置为0
        if self.args.cls_position == 'zero':
            position_ids = [0, 0] + [i + 1 for i in range(len(input_ids) - 2)]
        else:
            position_ids = [i for i in range(len(input_ids))]

        #分别标齐raw words和cand name的关键词位置
        raw_name_att = [0] * 2 + [1 if (raw_name_body_att[i] == 1) else 0 for i in range(len(raw_name_body_att))] \
                + [0] * (self.max_len - len(raw_name_body_att) - 2)
        cand_name_att = [0] * (3 + len(raw_name_body_att)) + [1 if (cand_name_body_att[i] == 1) else 0 for i in range(len(cand_name_body_att))]
        cand_name_att = cand_name_att[:self.max_len]
        cand_name_att += [0] * (self.max_len - len(cand_name_att))
        assert len(raw_name_att) == self.max_len, print(len(raw_name_att), len(raw_name_body_att))
        assert len(cand_name_att) == self.max_len

        #得到body和ot在encode后的位置信息，方便构造attention矩阵
        body_att_index = [i + 2 for i in range(len(raw_name_body_att)) if raw_name_body_att[i] == 1] + \
                            [i + 3 + len(raw_name_input_ids) for i in range(len(cand_name_body_att)) if cand_name_body_att[i] == 1]

        attention_mask = [[1, 0] + [1] * (len(input_ids) - 2) for _ in range(len(input_ids))]
        #body_cls只能看见所有的body词
        attention_mask[1] = [0, 1] + [1 if i in body_att_index else 0 for i in range(2, len(input_ids))]

        #手术词能够看到body cls, 同时对应的keywords attention也为1
        for i in body_att_index:
            # if i < self.max_len:
            attention_mask[i][1] = 1
        
        #进行padding
        for i in range(len(attention_mask)):
            attention_mask[i] += [0] * (self.max_len - len(attention_mask[i])) 
        
        attention_mask += [[0] * self.max_len] * (self.max_len - len(attention_mask))
        # print(attention_mask)
        assert len(attention_mask) == self.max_len
        input_ids += [0] * (self.max_len - len(input_ids))
        position_ids += [0] * (self.max_len - len(position_ids))
        token_type_ids += [0] * (self.max_len - len(token_type_ids))
        
        # print(input_ids)
        # print(position_ids)
        # print(token_type_ids)

        # print(raw_name)
        # print(raw_name_body_att)
        # print(raw_name_input_ids)
        # print(raw_name_att)

        # print(cand_name)
        # print(cand_name_body_att)
        # print(cand_name_input_ids)
        # print('1')
        # print(cand_name_att)
        # print('222')

        # for z in attention_mask:
        #     print(z)

        # input()


        return input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att

    def __getitem__(self, idx):
        raw_name, cand_name, label, cand_score = self.item_data_list[idx]

        # 构造coding type的label
        pos_name_types = [self.code_dict[self.name_to_code[x].split('.')[0]] for x in self.raw_name_pos_names[raw_name]]
        neg_type = self.code_dict[self.name_to_code[cand_name].split('.')[0]]
        code_type_label = 1 if neg_type in pos_name_types else 0

        #得到各自的att
        raw_name_body_att = self.key_words[raw_name]['body_att']
        # raw_name_ot_att = self.key_words[raw_name]['ot_att']
        cand_name_body_att = self.key_words[cand_name]['body_att']
        # cand_name_ot_att = self.key_words[cand_name]['ot_att']
        assert len(raw_name) == len(raw_name_body_att)
        # assert len(cand_name) == len(cand_name_ot_att)
        # 4是因为cls + 2个sep + body_cls
        if len(raw_name) + len(cand_name) + 4 > self.max_len:
            if len(raw_name) > len(cand_name):
                end_index = self.max_len - len(cand_name) - 4
                raw_name = raw_name[:end_index]  
                raw_name_body_att = raw_name_body_att[:end_index]
                # raw_name_ot_att = raw_name_ot_att[:end_index]
            else:
                end_index = self.max_len - len(raw_name) - 4
                cand_name = cand_name[:end_index]
                cand_name_body_att = cand_name_body_att[:end_index]
                # cand_name_ot_att = cand_name_ot_att[:end_index]
        input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att = self.tokenize(raw_name, cand_name, raw_name_body_att, cand_name_body_att)


        return input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, label, cand_score, code_type_label

    def collate_fn(self, batch):
        batch_item_len = len(batch[0])
        input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, label, cand_score, code_type_label = \
            [torch.tensor([item[x] for item in batch]).to(self.device) for x in range(batch_item_len)]

        return input_ids, attention_mask, position_ids, token_type_ids, raw_name_att, cand_name_att, code_type_label, label, cand_score
        
#构建val和test以及icd的dataset
class MultiClassDataset(Dataset):
    def __init__(self, data_list, tokenizer, args, standard_name_list, logger=None):
        self.data_list = data_list
        self.max_len = args.max_len
        self.batch_size = args.val_batch_size
        self.device = args.device
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger
        self.standard_name_list = standard_name_list
        

    def __getitem__(self, idx):
        raw_name, pos_name_list = self.data_list[idx]
        input_ids, attention_mask = tokenize(raw_name, self.tokenizer, self.max_len)
        
        pos_name_index_list = [self.standard_name_list.index(x) for x in pos_name_list]
        label = [0 for _ in range(len(self.standard_name_list))]
        for idx in pos_name_index_list:
            label[idx] = 1

        return input_ids, attention_mask, label, pos_name_list, raw_name
        

    def __len__(self):
        return len(self.data_list)

    def collate_fn(self, batch):
        batch_item_len = len(batch[0])
        input_ids, attention_mask, label, pos_name_list, raw_name = list(zip(*batch)) 
        input_ids = torch.tensor(input_ids).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)
        label = torch.tensor(label).to(self.device)

        return input_ids, attention_mask, label, pos_name_list, raw_name


        
