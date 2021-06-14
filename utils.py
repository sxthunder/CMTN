import pandas as pd 
import faiss
import numpy as np 
import os 
import transformers
import torch
import pickle
import random
from transformers import BertModel, BertTokenizer, BertConfig, BertLayer
from tqdm import tqdm
from transformers import AdamW
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
#读取训练数据以及code
def read_data(path, logger, args):
    logger.info('load data in {}'.format(path))
    train_data = np.array(pd.read_excel(os.path.join(path, 'train.xlsx'))).tolist()
    val_data = np.array(pd.read_excel(os.path.join(path, 'val.xlsx'))).tolist()
    answer_data = np.array(pd.read_excel(os.path.join(path, 'answer.xlsx'))).tolist()

    train_data = [(x[0], x[1].split('##')) for x in train_data]
    val_data = [(x[0], x[1].split('##')) for x in val_data]
    answer_data = [(x[0], x[1].split('##')) for x in answer_data]

    code_df = pd.read_csv(os.path.join(path, 'code.txt'), sep='\t', header=None)
    code_list = np.array(code_df).tolist()
    code_to_name = {x[0]:x[1] for x in code_list}
    name_to_code = {v:k for k, v in code_to_name.items()}
    name_list = list(name_to_code.keys())

    #emnlp2020中将answer和val进行了合并, 如果不是k—fold则送train set中割出一个部分
    answer_data = val_data + answer_data

    #划分得到train和val
    path = os.path.join(os.path.dirname(path), 'k_fold_data')
    train_data, val_data = split_train_and_val(path, train_data, args, logger)
    
    # 临时代码
    # answer_data = train_data + val_data
    print(len(train_data), len(val_data), len(answer_data))

    return train_data, val_data, answer_data, code_to_name, name_to_code, name_list

#根据参数划分train和val，如果不是k fold则是随机划分，否则就是k fold划分
def split_train_and_val(path, train_data, args, logger):
    if args.k_fold == -1:
        val_data = train_data[3500:]
        train_data = train_data[:3500]

    else:
        #如果没有划分k_fold, 则划分k_fold
        print(os.path.exists(path))
        if not os.path.exists(path):
            logger.info('create k fold data in {}'.format(path))
            os.makedirs(path)

            kf = KFold(n_splits=5, shuffle=True)
            for idx, (train_index, val_index) in enumerate(kf.split(train_data)):
                x_train = [train_data[x] for x in train_index]
                x_val = [train_data[x] for x in val_index]
                pickle.dump(x_train, open(os.path.join(path, 'x_train_{}'.format(idx)), 'wb'))
                pickle.dump(x_val, open(os.path.join(path, 'x_val_{}'.format(idx)), 'wb'))
        logger.info('load {} fold data from {}'.format(args.k_fold, path))
        train_data = pickle.load(open(os.path.join(path, 'x_train_{}'.format(args.k_fold)), 'rb'))
        val_data = pickle.load(open(os.path.join(path, 'x_val_{}'.format(args.k_fold)), 'rb'))
    return train_data, val_data

def read_rerank_data(path, logger, args):
    logger.info('load data from {}'.format(path))
    train_data = pickle.load(open(os.path.join(path, 'train_candidate_list'), 'rb'))
    test_data = pickle.load(open(os.path.join(path, 'test_candidate_list'), 'rb'))

    standard_path = os.path.join(os.path.dirname(path), 'data')
    code_df = pd.read_csv(os.path.join(standard_path, 'code.txt'), sep='\t', header=None)
    code_list = np.array(code_df).tolist()
    code_to_name = {x[0]:x[1] for x in code_list}
    name_to_code = {v:k for k, v in code_to_name.items()}
    name_list = list(name_to_code.keys())

    k_fold_path = os.path.join(os.path.dirname(path), 'rerank_k_fold_data')
    # k_fold_path = os.path.join(os.path.dirname(path), 'rerank_tf_idf_k_fold_data')
    train_data, val_data = split_train_and_val(k_fold_path, train_data, args, logger)
    print(len(train_data), len(val_data), len(test_data))

    return train_data, val_data, test_data, code_to_name, name_to_code, name_list

def get_pretrained_model(path, logger, args=None):
    logger.info('load pretrained model in {}'.format(path))
    bert_tokenizer = BertTokenizer.from_pretrained(path)
    
    if args is None or args.hidden_layers == 12:
        bert_config = BertConfig.from_pretrained(path)
        bert_model = BertModel.from_pretrained(path)

    else:
        logger.info('load {} layers bert'.format(args.hidden_layers))
        bert_config = BertConfig.from_pretrained(path, num_hidden_layers=args.hidden_layers)
        bert_model = BertModel(bert_config)
        model_param_list = [p[0] for p in bert_model.named_parameters()]
        load_dict = torch.load(os.path.join(path, 'pytorch_model.bin'))
        new_load_dict = {}
        for k, v in load_dict.items():
            k = k.replace('bert.', '')
            if k in model_param_list:
                new_load_dict[k] = v
        new_load_dict['embeddings.position_ids'] = torch.tensor([i for i in range(512)]).unsqueeze(dim=0)
        bert_model.load_state_dict(new_load_dict)

    logger.info('load complete')
    return bert_config, bert_tokenizer, bert_model

def print_args(args, logger):
    logger.info('#'*20 + 'Arguments' + '#'*20)
    arg_dict = vars(args)
    for k, v in arg_dict.items():
        logger.info('{}:{}'.format(k, v))

def tokenize(text, tokenizer, max_len):
    if len(text) + 2 > max_len:
        text = text[:max_len - 2]
    token_result = tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', max_length=max_len)
    input_ids = token_result['input_ids']
    attention_mask = token_result['attention_mask']
    assert len(input_ids) == max_len
    assert len(attention_mask) == max_len
    return input_ids, attention_mask

def get_optimizer_and_scheduler(model, t_total, lr, warmup_steps, eps=1e-6, optimizer_class=AdamW, scheduler='WarmupLinear'):
    def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01, 'lr':lr},
        {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay))], 'weight_decay': 0.0, 'lr':lr},
    ]

    local_rank = -1
    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    optimizer_params = {'lr': lr, 'eps': eps}
    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
#     scheduler_obj = get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=t_total)
    scheduler_obj = None
    return optimizer, scheduler_obj

# 通过模型预测得到icd的标准词或者测试集和的embedidng(如果是测试集合会多一些输出)
# 如果是icd则只返回embedding，如果是k_fold会多返回pred score，如果需要return att则返回keywords attention
def get_model_embedding(model, dataloader, is_icd=False, is_k_fold=False):
    pred_list = []
    pred_score_list = []
    embedidng_list = []
    pos_name_list = []
    raw_name_list = []
    label_list = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dataloader):
            input_ids, attention_mask = batch[:2]
            cls_logits, seq_pool_output = model.get_bert_output(input_ids, attention_mask)
            embedidng_list += seq_pool_output.cpu().tolist()
            if not is_icd:
                pred_list += cls_logits.argmax(dim=-1).cpu().tolist()
                batch_pos_name_list, batch_raw_name_list, pos_len = batch[2:5]
                pos_name_list += batch_pos_name_list
                raw_name_list += batch_raw_name_list
                # label_list += pos_len.cpu().tolist()
                label_list += [len(x) - 1 for x in batch_pos_name_list]
                # k fold 的话需要返回分类预测的概率
                if is_k_fold:
                    pred_score_list += cls_logits.cpu().tolist()

    embedidng_list = np.array(embedidng_list)
    if not is_icd:
        if is_k_fold:
            return embedidng_list, pos_name_list, raw_name_list, pred_list, label_list, pred_score_list
        else:
            return embedidng_list, pos_name_list, raw_name_list, pred_list, label_list
    else:
        return embedidng_list

#在online 生成新的train_dataloader时,使用faiss查找最近的neg_num+1个即可
def quick_search_neighbor(query_embed, doc_embed, args, k=10):
    d = query_embed.shape[1]

    if args.distance == 'eu':
        index = faiss.IndexFlatL2(d)
    else:
        #如果是cosine,先做归一化再用indexflatip
        query_norm = np.linalg.norm(query_embed, axis=1, keepdims=True)
        doc_norm = np.linalg.norm(doc_embed, axis=1, keepdims=True)
        query_embed = query_embed / query_norm
        doc_embed = doc_embed / doc_norm
        index = faiss.IndexFlatIP(d)

    doc_embed = doc_embed.astype(np.float32)
    query_embed = query_embed.astype(np.float32)
    index.add(doc_embed)
    _, I = index.search(query_embed, k)

    #len(query_embed), k
    return I

def get_similarity_matrix(a, b, distance_mode):
    #欧式距离或1-cosine
    if distance_mode == 'eu':
        return cdist(a, b, p=2)
    else:
        return cdist(a, b, 'cosine')

def get_train_neighbor(model, train_dataloader, icd_embedding, args, logger, train_list, standard_name_list):
    #生成train标准词的最新embedding
    logger.info('generate training set embedding')
    train_embedding = get_model_embedding(model, train_dataloader, True) 
    train_index = quick_search_neighbor(train_embedding, icd_embedding, args, k=15)

    neg_list = []
    for (raw_name, pos_name_list), neighbor_index in zip(train_list, train_index):
        neighbor_list = [standard_name_list[x] for x in neighbor_index if standard_name_list[x] not in pos_name_list]
        #这里因为每个posname其实对应相同的neg，就直接copy出来即可
        neighbor_list = neighbor_list[:args.neg_num]
        assert len(neighbor_list) == args.neg_num
        neg_list.append([neighbor_list for _ in range(len(pos_name_list))])

    assert len(neg_list) == len(train_list)
    return neg_list

#读取字典
def read_dict(path):
    l = []
    with open(path, 'r') as f:
        for line in f:
            l.append(line.replace('\n', '').split('\t')[0])
        f.close()
    return list(set(l))

#rerank进行evaluate
def rerank_evaluate(dataloader, model, logger):
    y_pred = []
    y_pred_score = []
    y_true = []
    cand_score = []
    model.eval()
    logger.info('evaluate for classification')
    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits = model.forward(batch[:-1])
            y_pred += torch.argmax(logits, dim=-1).cpu().tolist()
            y_pred_score += torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            y_true += batch[-2].cpu().tolist()
            cand_score += batch[-1].cpu().tolist()
    return y_pred, y_pred_score, y_true, cand_score

#rerank coding进行evaluate
def rerank_coding_evaluate(dataloader, model, logger):
    cls_pred = []
    cls_score = []
    cls_true = []
    kw_pred = []
    kw_score = []
    kw_true = []

    cand_score = []

    model.eval()
    logger.info('evaluate for classification')
    with torch.no_grad():
        for batch in tqdm(dataloader):
            cls_logits, kw_logits = model.forward(batch[:-1])

            cls_pred += torch.argmax(cls_logits, dim=-1).cpu().tolist()
            cls_score += torch.softmax(cls_logits, dim=-1)[:, 1].cpu().tolist()
            cls_true += batch[-2].cpu().tolist()

            kw_pred += torch.argmax(kw_logits, dim=-1).cpu().tolist()
            kw_score += torch.softmax(kw_logits, dim=-1)[:, 1].cpu().tolist()
            kw_true += batch[-3].cpu().tolist()

            cand_score += batch[-1].cpu().tolist()
    return cls_pred, cls_score, cls_true, kw_pred, kw_score, kw_true, cand_score

# rerank coding有两个similarity分数，采取不同的策略进行融合
def rerank_coding_pred_merge(cls_pred_score, kw_pred_score, args):
    cls_pred_score = np.array(cls_pred_score)
    kw_pred_score = np.array(kw_pred_score)
    # 如果theta大于零，只有cls和kw的分数都大于theta才按照下面的方式进行融合，否则直接记为0
    if args.theta > 0:
        b = (cls_pred_score > args.theta) * (kw_pred_score > args.theta)
        cls_pred_score = b * cls_pred_score
        kw_pred_score = b * kw_pred_score

    #两个分数直接乘起来
    if args.code_merge == 'product':
        print('merge code by product')
        return (cls_pred_score * kw_pred_score).tolist()

    elif args.code_merge == 'add':
        print('merge code by add')
        return (cls_pred_score + kw_pred_score).tolist()


# rerank的时候构造各种中间变量，以使其使用metric函数进行预测
# ratio是cand score和rerank score融合时，cand score所占的比例，默认0.5均分
def rerank_for_metric(dataloader, y_pred_score, cand_score, standard_name_list, args, ratio=0.5):
    # 构造similarity matrix和similarity score等变量，直接带入metric函数进行预测
    similarity_matrix = []
    similarity_score = [] #因为不需要生成candidate，这个数组其实意义不大，不过要凑成相似的形状放进去
    label_list = []
    pred_list = []
    raw_name_list = []
    pos_name_list = []
    raw_name_sim = []
    raw_name_cand_score = []
    last_raw_name = ''
    for item, pred_score in zip(dataloader.dataset.item_data_list, y_pred_score):
        raw_name, cand_name, label, cand_score = item
        if last_raw_name != '' and last_raw_name != raw_name:
            raw_name_list.append(last_raw_name)
            pos_name_list.append(dataloader.dataset.raw_name_pos_names[last_raw_name])

            label_list.append(dataloader.dataset.raw_name_num_label[last_raw_name] - 1)
            pred_list.append(dataloader.dataset.raw_name_num_pred[last_raw_name] - 1)
            
            #如果需要和之前的模型进行融合,且当前这个sample是一对一 时才可以进行融合
            if args.merge_with_bert_sort == 'yes':
                assert len(raw_name_cand_score) == len(raw_name_sim)
                raw_name_cand_score = 1 - (np.array(raw_name_cand_score) / max(raw_name_cand_score))
                for i in range(len(raw_name_sim)):
                    idx, rerank_score = raw_name_sim[i]
                    raw_name_sim[i] = (idx, (1 - ratio) * rerank_score + ratio * raw_name_cand_score[i])
                    # raw_name_sim[i] = (idx, rerank_score)

            raw_name_sim_index = [y[0] for y in sorted(raw_name_sim, key=lambda x:-x[1])]
            similarity_matrix.append(raw_name_sim_index)
            similarity_score.append([])
            raw_name_sim = []
            raw_name_cand_score = []

        raw_name_sim.append((standard_name_list.index(cand_name), pred_score))
        raw_name_cand_score.append(cand_score)
        last_raw_name = raw_name

    raw_name_list.append(last_raw_name)
    pos_name_list.append(dataloader.dataset.raw_name_pos_names[last_raw_name])
    raw_name_sim_index = [y[0] for y in sorted(raw_name_sim, key=lambda x:x[1])]
    similarity_matrix.append(raw_name_sim_index)
    similarity_score.append([])
    label_list.append(dataloader.dataset.raw_name_num_label[last_raw_name] -1) #分类任务，下标从0开始，因此实际的label比真是长度小1
    pred_list.append(dataloader.dataset.raw_name_num_pred[last_raw_name] -1)
    return raw_name_list, pos_name_list, similarity_matrix, similarity_score, label_list, pred_list

# 对keywords attention进行padding
def padding_keywords_att(att, max_len):
    if len(att) > max_len - 2:
        att = att[:max_len-2]
    att = [0] + att + [0] * (max_len - len(att) - 1)
    assert len(att) == max_len
    return att

# def get_keywords_att(raw_name, )
# 读取keywords的信息
def load_keywords(logger):
    def load(path):
        df = pd.read_excel(path)
        df.fillna('', inplace=True)
        d = {}
        for index, row in df.iterrows():
            term = row['term']
            body = row['body'].split(',') if row['body'] != '' else []
            body_att = [int(x) for x in row['body_attention'].split(',')]
            rulu = row['rulu'].split(',') if row['rulu'] != '' else []
            rulu_att = [int(x) for x in row['rulu_attention'].split(',')]
            ot = row['ot'].split(',') if row['ot'] != '' else []
            ot_att = [int(x) for x in row['ot_attention'].split(',')] 

            d[term] = {'body':body, 'body_att':body_att, 'rulu':rulu, 'rulu_att':rulu_att, 'ot':ot, 'ot_att':ot_att}
        return d
    logger.info('loading keywords.....')
    body = read_dict('/home/liangming/nas/ml_project/terminology_normalization/chip2019/dict/body.txt')
    rulu = read_dict('/home/liangming/nas/ml_project/terminology_normalization/chip2019/dict/rulu.txt')
    ot = read_dict('/home/liangming/nas/ml_project/terminology_normalization/chip2019/dict/ot.txt')
    d = {'body':body, 'rulu':rulu, 'ot':ot}

    key_words = load('/home/liangming/nas/ml_project/terminology_normalization/chip2019/dict_match_result/train.xlsx')
    key_words.update(load('/home/liangming/nas/ml_project/terminology_normalization/chip2019/dict_match_result/val.xlsx'))
    key_words.update(load('/home/liangming/nas/ml_project/terminology_normalization/chip2019/dict_match_result/name.xlsx'))
    key_words.update(load('/home/liangming/nas/ml_project/terminology_normalization/chip2019/dict_match_result/answer.xlsx'))
    return d, key_words

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_coding_type():
    dict_list = [
        [0],
        [1, 5],
        [6, 7],
        [8, 16],
        [17],
        [18, 20],
        [21, 29],
        [30, 34],
        [35, 39],
        [40, 41],
        [42, 54],
        [55, 59],
        [60, 64],
        [65, 71],
        [72, 75],
        [76, 84],
        [85, 86],
        [87, 99]
    ]
    d = {}
    for idx, item in enumerate(dict_list):
        for i in range(item[0], item[-1] + 1):
            if i < 10:
                key = '0' + str(i)
            else:
                key = str(i)
            d[key] = idx
    return d