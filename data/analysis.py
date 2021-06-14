import numpy as np 
import pandas as pd  
import os
def label_distribution(l):
    uni_count = 0
    for item in l:
        item_len = item[1].split('##')
        if len(item_len) == 1:
            uni_count += 1
            assert '#' not in item[1], print(item[1])
        else:
            pass
    print('uni count {}, multi count {}'.format(uni_count, len(l) - uni_count))

code_df = pd.read_csv('./code.txt', sep='\t', header=None)
code_list = np.array(code_df).tolist()
code_to_name = {x[0]:x[1] for x in code_list}
name_to_code = {v:k for k, v in code_to_name.items()}

print('code to name len is {}, name to code len is {}'.format(len(code_to_name), len(name_to_code)))

train_data = np.array(pd.read_excel('./train.xlsx')).tolist()
val_data = np.array(pd.read_excel('./val.xlsx')).tolist()
answer_data = np.array(pd.read_excel('./answer.xlsx')).tolist()

print(len(train_data), len(val_data), len(answer_data))
# label_distribution(train_data)
label_distribution(val_data)
label_distribution(answer_data)

