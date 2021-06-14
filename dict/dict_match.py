import pandas as pd 
import numpy as np
import os

def read_dict(path):
    l = []
    with open(path, 'r') as f:
        for line in f:
            l.append(line.replace('\n', '').split('\t')[0])
        f.close()
    return list(set(l))

def read_data(path):
    train_data = np.array(pd.read_excel(os.path.join(path, 'train.xlsx'))).tolist()
    val_data = np.array(pd.read_excel(os.path.join(path, 'val.xlsx'))).tolist()
    answer_data = np.array(pd.read_excel(os.path.join(path, 'answer.xlsx'))).tolist()

    train_data = [x[0] for x in train_data]
    val_data = [x[0] for x in val_data]
    answer_data = [x[0] for x in answer_data]

    code_df = pd.read_csv(os.path.join(path, 'code.txt'), sep='\t', header=None)
    code_list = np.array(code_df).tolist()
    name_list = [x[1] for x in code_list]
    code_to_name = {x[0]:x[1] for x in code_list}
    name_to_code = {v:k for k, v in code_to_name.items()}
    return train_data, val_data, answer_data, name_list

def dict_match(body_list, rulu_list, ot_list, l, file_name):
    print(file_name + '{}'.format(len(l)))
    def match(dl, item, is_ot=False):
        # temp = sorted(, key=lambda x:-len(x))
        temp = list(set([x for x in dl if x in item])) 

        # 如果是术式，可能term中是 切开、xxxx术，这种也算切开术
        if is_ot:
            temp += [x for x in dl if x[:2] in item]
            temp = list(set(temp))

        new_temp = []
        for t in temp:
            flag = True
            for nt in temp:
                if t == nt:
                    continue
                if t in nt:
                    flag = False
                    break
            if flag:
                new_temp.append(t)
        
        match_attention = ['0'] * len(item)
        if len(temp) == 0:
            return '', ','.join(match_attention)

        for t in new_temp:
            if t not in item:
                t = t[:-1]
            start = item.index(t)
            end = start + len(t)
            match_attention[start:end] = ['1'] * len(t)
        return ','.join(new_temp), ','.join(match_attention)
    mr = []
    for item in l:
        body, body_attention = match(body_list, item)
        rulu, rulu_attention = match(rulu_list, item)
        ot, ot_attention = match(ot_list, item, is_ot=True)
        mr.append([item, body, body_attention, rulu, rulu_attention, ot, ot_attention])

    _, body, _,  rulu, _, ot, _ = list(zip(*mr))
    print('body matched {}'.format(len([x for x in body if x != ''])))
    print('rulu matched {}'.format(len([x for x in rulu if x != ''])))
    print('ot matched {}'.format(len([x for x in ot if x != ''])))

    c = ['term', 'body', 'body_attention', 'rulu', 'rulu_attention', 'ot', 'ot_attention']
    df = pd.DataFrame(mr, columns=c)
    df.to_excel('../dict_match_result/{}.xlsx'.format(file_name))

body_list = read_dict('./body.txt')
rulu_list = read_dict('./rulu.txt')
ot_list = read_dict('./ot.txt')
train_data, val_data, answer_data, name_list = read_data('/home/liangming/nas/ml_project/terminology_normalization/chip2019/data')

dict_match(body_list, rulu_list, ot_list, train_data, 'train')
dict_match(body_list, rulu_list, ot_list, val_data, 'val')
dict_match(body_list, rulu_list, ot_list, answer_data, 'answer')
dict_match(body_list, rulu_list, ot_list, name_list, 'name')



