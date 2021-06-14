import os
import time
import numpy as np
import torch
import pickle
import time
from utils import read_data, get_pretrained_model, print_args, get_optimizer_and_scheduler, get_model_embedding, quick_search_neighbor, get_similarity_matrix, get_train_neighbor, set_seed
from argparse import ArgumentParser
from dataset import TripleDataset, TermDataset
from log import Logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import SiameseClassificationModel
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from collections import Counter
project_path = '/home/liangming/nas/ml_project/terminology_normalization/chip2019'

def main():
    parser = ArgumentParser()

    #任务配置
    parser.add_argument('-neg_sample', default='online', type=str)
    parser.add_argument('-neg_num', default=4, type=int)
    parser.add_argument('-device', default=0, type=int)
    parser.add_argument('-output_name', default='', type=str)
    parser.add_argument('-saved_model_path', default='', type=str) #如果是k fold合并模型进行预测，只需设置为对应k_fold模型对应的output path
    parser.add_argument('-type', default='train', type=str)
    parser.add_argument('-k_fold', default=-1, type=int) #如果是-1则说明不采用k折，否则说明采用k折的第几折
    parser.add_argument('-merge_classification', default='vote', type=str) # 个数预测：vote则采用投票法，avg则是平均概率
    parser.add_argument('-merge_sort', default='avg', type=str) # k折融合算法: vote则采用投票法，avg则是平均概率
    parser.add_argument('-generate_candidates', default='no', type=str) # 在预测过程中，是否生成候选词并保存在output_name下，如果是no则不生成，如果不是no则是保存的名字
    parser.add_argument('-k_fold_cache', default='no', type=str) # 在预测k_fold中，是否采用之前的cache，如果存在cache，会在output_name下存在三个list文件, 避免5个模型进行预测
    parser.add_argument('-add_keywords', default='no', type=str) # 这里不添加关键词信息
    parser.add_argument('-seed', default=123456, type=int) # 种子
    parser.add_argument('-loss_type', default='union', type=str) # loss type: class就仅更新分类的参数；sim就仅更新triple loss；union就同时更新二者参数
    parser.add_argument('-hidden_layers', default=12, type=int) # bert的层数

    #训练参数
    parser.add_argument('-train_batch_size', default=64, type=int)
    parser.add_argument('-val_batch_size', default=256, type=int)
    parser.add_argument('-lr', default=2e-5, type=float)
    parser.add_argument('-epoch_num', default=20, type=int)

    parser.add_argument('-max_len', default=64, type=int)
    parser.add_argument('-margin', default=1, type=float)
    parser.add_argument('-distance', default='eu', type=str)
    parser.add_argument('-label_nums', default=3, type=int)
    parser.add_argument('-dropout', default=0.3, type=float)
    parser.add_argument('-pool', default='avg', type=str)
    parser.add_argument('-print_loss_step', default=2, type=int)
    parser.add_argument('-hit_list', default=[2, 5, 7, 10], type=list)


    args = parser.parse_args()
    assert args.train_batch_size % args.neg_num == 0, print('batch size应该是neg_num的整数倍')

    #定义时间格式
    DATE_FORMAT = "%Y-%m-%d-%H:%M:%S"
    #定义输出文件夹，如果不存在则创建, 
    if args.output_name == '':
        output_path = os.path.join('./output/mto_output', time.strftime(DATE_FORMAT,time.localtime(time.time())))
    else:
        output_path = os.path.join('./output/mto_output', args.output_name)
        # if os.path.exists(output_path):
            # raise Exception('the output path {} already exists'.format(output_path))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #配置tensorboard    
    tensor_board_log_path = os.path.join(output_path, 'tensor_board_log{}'.format('' if args.k_fold == -1 else args.k_fold))
    writer = SummaryWriter(tensor_board_log_path)

    #定义log参数
    logger = Logger(output_path,'main{}'.format('' if args.k_fold == -1 else args.k_fold)).logger
    
    #打印args
    print_args(args, logger)

    #设置seed
    logger.info('set seed to {}'.format(args.seed))
    set_seed(args)

    #读取数据
    logger.info('#' * 20 + 'loading data and model' + '#' * 20)
    data_path = os.path.join(project_path, 'data')
    train_list, val_list, test_list, code_to_name, name_to_code, standard_name_list = read_data(data_path, logger, args)

    #load model
    pretrained_model_path = '/home/liangming/nas/lm_params/chinese_L-12_H-768_A-12/'
    bert_config, bert_tokenizer, bert_model = get_pretrained_model(pretrained_model_path, logger, args)

    #获取dataset
    logger.info('create dataloader')
    train_dataset = TripleDataset(train_list, standard_name_list, code_to_name, name_to_code, bert_tokenizer, args, logger)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)

    val_dataset = TermDataset(val_list, bert_tokenizer, args, False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    test_dataset = TermDataset(test_list, bert_tokenizer, args, False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    icd_dataset = TermDataset(standard_name_list, bert_tokenizer, args, True)
    icd_dataloader = DataLoader(icd_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=icd_dataset.collate_fn)

    train_term_dataset = TermDataset(train_list, bert_tokenizer, args, False)
    train_term_dataloader = DataLoader(train_term_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=train_term_dataset.collate_fn)


    #创建model
    logger.info('create model')
    model = SiameseClassificationModel(bert_model, bert_config, args)
    model = model.to(args.device)

    #配置optimizer和scheduler
    t_total = len(train_dataloader) * args.epoch_num
    optimizer, _ = get_optimizer_and_scheduler(model, t_total, args.lr, 0)

    if args.type == 'train':
        train(model, train_dataloader, val_dataloader, test_dataloader, icd_dataloader, train_term_dataloader, optimizer, writer, args, logger, \
                standard_name_list, train_list, output_path, bert_tokenizer, name_to_code, code_to_name)

    elif args.type == 'evaluate':
        if args.saved_model_path == '':
            raise Exception('saved model path不能为空')

        # 非k折模型
        if args.k_fold == -1:
            logger.info('loading saved model')
            checkpoint = torch.load(args.saved_model_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            model = model.to(args.device)
            # #生成icd标准词的最新embedding
            logger.info('generate icd embedding')
            icd_embedding = get_model_embedding(model, icd_dataloader, True)
            start_time = time.time()
            evaluate(model, test_dataloader, icd_embedding, args, logger, writer, standard_name_list, True)
            end_time = time.time()
            logger.info('total length is {}, predict time is {}, per mention time is {}'.format(len(test_list), end_time - start_time, (end_time - start_time) / len(test_list)))

        else:
            evaluate_k_fold(model, test_dataloader, icd_dataloader, args, logger, writer, standard_name_list)
            # evaluate_k_fold(model, train_dataloader, icd_dataloader, args, logger, writer, standard_name_list)

def train(model, train_dataloader, val_dataloader, test_dataloader, icd_dataloader, train_term_dataloader, optimizer, writer, args, logger, \
            standard_name_list, train_list, output_path, bert_tokenizer, name_to_code, code_to_name):
    model.train()
    loss_list, c_loss_list, t_loss_list, c_acc_list = [], [], [], []
    best_acc_with_pred_label, best_acc_with_laebl, best_class_acc = 0, 0, 0
    step = 0
    model_saved_path = os.path.join(output_path, 'saved_model{}'.format('' if args.k_fold == -1 else args.k_fold))
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)

    for epoch in range(args.epoch_num):
        logger.info('#'*20 + 'Epoch{}'.format(epoch + 1) + '#'*20)
        iteration = tqdm(train_dataloader, desc='Training')
        for batch in iteration:
            model.zero_grad()
            tl, cl, ca = model.forward(batch)
            total_loss = tl + cl
            loss_list.append(total_loss.item())
            t_loss_list.append(tl.item())
            c_loss_list.append(cl.item())
            c_acc_list.append(ca.item())

            writer.add_scalar('total_loss', loss_list[-1], step)
            writer.add_scalar('triple_loss', t_loss_list[-1],step)
            writer.add_scalar('classification_loss', c_loss_list[-1], step)
            writer.add_scalar('classification_acc', c_acc_list[-1], step)
            if (step + 1) % args.print_loss_step == 0:
                iteration.set_description(
                'total loss:{}, class loss:{}, triple loss:{}, class acc:{}'.format(
                    round(sum(loss_list) / len(loss_list), 4),
                    round(sum(c_loss_list) / len(c_loss_list), 4),
                    round(sum(t_loss_list) / len(t_loss_list), 4),
                    round(sum(c_acc_list) / len(c_acc_list), 4)))
            if args.loss_type == 'class':
                cl.backward()

            elif args.loss_type == 'triple':
                tl.backward()

            elif args.loss_type == 'union':
                total_loss.backward()
            # tl.backward()
            optimizer.step()

            step += 1

        # #生成icd标准词的最新embedding
        logger.info('generate icd embedding')
        icd_embedding = get_model_embedding(model, icd_dataloader, is_icd=True)

        logger.info('#'*20 + 'Evaluate' + '#'*20)
        acc_with_pred_label, acc_with_label, class_acc = evaluate(model, val_dataloader, icd_embedding, args, logger, writer, standard_name_list)

        if acc_with_pred_label > best_acc_with_pred_label:
            best_acc_with_pred_label = acc_with_pred_label
            logger.info('save model_with_pred_label in acc {}'.format(best_acc_with_pred_label))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'model_with_pred_label.pth'))
        
        if acc_with_label > best_acc_with_laebl:
            best_acc_with_laebl = acc_with_label
            logger.info('save model with label in acc {}'.format(best_acc_with_laebl))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'model_with_label.pth'))

        if class_acc > best_class_acc:
            best_class_acc = class_acc 
            logger.info('save model with best classification in acc {}'.format(best_class_acc))
            torch.save(model.state_dict(), os.path.join(model_saved_path, 'classification_model.pth'))

        logger.info('#'*20 + 'Test' + '#'*20)
        evaluate(model, test_dataloader, icd_embedding, args, logger, writer, standard_name_list, is_test=True)

        #判断是否为online生成负例的策略
        if args.neg_sample == 'online':
            train_neg_list = get_train_neighbor(model, train_term_dataloader, icd_embedding, args, logger, train_list, standard_name_list)
            train_dataset = TripleDataset(train_list, standard_name_list, code_to_name, name_to_code, bert_tokenizer, args, logger, train_neg_list)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=train_dataset.collate_fn)
def evaluate_k_fold(model, dataloader, icd_dataloader, args, logger, writer, standard_name_list):
    logger.info('#'*20 + 'evluate k fold' + '#' * 20)
    logger.info('merge k fold model to evaluate')
    #公共数据在外面假装定义一下
    raw_name_list = []
    pos_name_list = []
    label_list = []

    #不采用cache
    if args.k_fold_cache == 'no':
        # 用于保存每个fold的similarity score matrix（未排序版本)
        k_fold_similarity_score_list = []
        # 用于保存每个fold的个数预测情况,包括label以及对应的score
        k_fold_pred_list = []
        k_fold_pred_score_list = []
        for i in range(5):
            saved_model_path = os.path.join(args.saved_model_path, 'saved_model{}'.format(i), 'model_with_pred_label.pth')
            logger.info('loading {} fold model from {}'.format(i, saved_model_path))
            checkpoint = torch.load(saved_model_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            model = model.to(args.device)
            # #生成icd标准词的最新embedding
            logger.info('generate icd embedding')
            icd_embedding = get_model_embedding(model, icd_dataloader, True)

            # 预测当前的一堆embedding
            logger.info('generate query embedding')
            query_embedding, pos_name_list, raw_name_list, pred_list, label_list, pred_score_list = get_model_embedding(model, dataloader, is_icd=False, is_k_fold=True)
            print(query_embedding.shape)
            # break
            k_fold_pred_list.append(pred_list)
            k_fold_pred_score_list.append(pred_score_list)
            
            #这里只保留分数，不做排序
            logger.info('generate similarity matrix')
            similarity_matrix = get_similarity_matrix(query_embedding, icd_embedding, args.distance)
            assert len(pred_list) == len(similarity_matrix)

            k_fold_similarity_score_list.append(similarity_matrix)

        pickle.dump(k_fold_pred_list, open('./output/mto_output/{}/{}_k_fold_pred_list'.format(args.output_name, args.generate_candidates), 'wb'))
        pickle.dump(k_fold_pred_score_list, open('./output/mto_output/{}/{}_k_fold_pred_score_list'.format(args.output_name, args.generate_candidates), 'wb'))
        pickle.dump(k_fold_similarity_score_list, open('./output/mto_output/{}/{}_k_fold_similarity_score_list'.format(args.output_name, args.generate_candidates), 'wb'))

    else:
        logger.info('load k_fold cache')
        _, pos_name_list, raw_name_list, _, label_list, _ = get_model_embedding(model, dataloader, is_icd=False, is_k_fold=True)
        k_fold_pred_list = pickle.load(open('./output/mto_output/{}/{}_k_fold_pred_list'.format(args.output_name, args.generate_candidates), 'rb'))
        k_fold_pred_score_list = pickle.load(open('./output/mto_output/{}/{}_k_fold_pred_score_list'.format(args.output_name, args.generate_candidates), 'rb'))
        k_fold_similarity_score_list = pickle.load(open('./output/mto_output/{}/{}_k_fold_similarity_score_list'.format(args.output_name, args.generate_candidates), 'rb'))
        print('./output/{}/{}_k_fold_pred_list'.format(args.output_name, args.generate_candidates))
    pred_list, similarity_matrix, similarity_score = merge_k_fold_result(k_fold_pred_list, k_fold_pred_score_list, k_fold_similarity_score_list, args, logger)
    metric(label_list, pred_list, similarity_matrix, similarity_score, args, raw_name_list, pos_name_list, standard_name_list, logger, False)

def merge_k_fold_result(k_fold_pred_list, k_fold_pred_score_list, k_fold_similarity_score_list, args, logger):
    # 首先合并个数预测
    k_fold_pred = np.array(k_fold_pred_list)
    k_fold_pred_score = np.array(k_fold_pred_score_list)
    k_fold_similarity_score = np.array(k_fold_similarity_score_list)

    test_len = len(k_fold_pred[0])
    assert k_fold_pred.shape == (5, test_len)
    assert k_fold_pred_score.shape == (5, test_len, 3)

    logger.info('merge classification by {}'.format(args.merge_classification))
    if args.merge_classification == 'vote':
        one_hot_pred = np.eye(3)[k_fold_pred] # 5, test_len, 3
        pred_list = np.argmax(np.sum(one_hot_pred, axis=0), axis=-1).tolist() # test_len
    elif args.merge_classification == 'avg':
        pred_list = np.argmax(np.sum(k_fold_pred_score, axis=0), axis=-1).tolist() # test_len

    logger.info('merge similarity by {}'.format(args.merge_sort))
    if args.merge_sort == 'avg':
        similarity_score = np.sum(k_fold_similarity_score, axis=0)
        similarity_matrix = np.argsort(similarity_score)
    elif args.merge_sort == 'vote':
        logger.info('sort similarity score')
        # similarity_index = np.argsort(k_fold_similarity_score, axis=-1)[:, :, :20] #每个test item只保留最近的20个item
        # 随便定义一个空的，免得没有啥东西返回
        similarity_score = []
        similarity_index = np.argsort(k_fold_similarity_score, axis=-1) #每个test item只保留最近的20个item

        similarity_matrix = []
        for i in range(test_len):
            vote_pool = similarity_index[:, i, :].reshape(-1, )
            # similarity_matrix.append(list(Counter(vote_pool).keys())[:20])
            similarity_matrix.append(list(Counter(vote_pool).keys()))
    
    return pred_list, similarity_matrix, similarity_score

def evaluate(model, dataloader, icd_embedding, args, logger, writer, standard_name_list, is_test=False):
    logger.info('generate val set embedding')
    query_embedding, pos_name_list, raw_name_list, pred_list, label_list = get_model_embedding(model, dataloader, is_icd=False)

    logger.info('generate similarity matrix')
    # 如果测试的话，直接用fassi
    if is_test:
        similarity_matrix = quick_search_neighbor(query_embedding, icd_embedding, args)

    else:
        similarity_matrix = get_similarity_matrix(query_embedding, icd_embedding, args.distance)
        # keyword_similarity_matrix 
        logger.info('sort similarity matrix')
        similarity_matrix = similarity_matrix.argsort(axis=-1)
        assert len(pred_list) == len(similarity_matrix)

    similarity_score = [[] for _ in range(len(similarity_matrix))]
    acc_with_pred_label, acc_with_label, total_acc = metric(label_list, pred_list, similarity_matrix, similarity_score, args, raw_name_list, pos_name_list, standard_name_list, logger, is_test)

    return acc_with_pred_label, acc_with_label, total_acc

# 进行各种指标的统计, similarity_matrix 为每个item的最近的similarity index，len(test_list) 行, 列数不一定满
def metric(label_list, pred_list, similarity_matrix, similarity_score, args, raw_name_list, pos_name_list, standard_name_list, logger, is_test):
    def get_acc_count(l1, l2):
        if len(l1) == len(l2) and len(set(l1).union(set(l2))) == len(l1):
            return 1
        return 0
    report = classification_report(label_list, pred_list, digits=4)
    logger.info(report)
    #统计个数分类的情况，得到以下指标：
    #1.总体的acc
    #2.个数为1的acc
    #3.个数为多个的acc
    report_dict = classification_report(label_list, pred_list, digits=4, output_dict=True)
    total_acc = report_dict['accuracy']
    one_acc = report_dict['0']['recall']
    multi_acc = (report_dict['1']['recall'] + report_dict['2']['recall']) / 2
    logger.info('classification: total acc is {}'.format(round(total_acc, 4)))
    logger.info('classification: one acc is {}'.format(round(one_acc, 4)))
    logger.info('classification: multi acc is {}'.format(round(multi_acc, 4)))


    #开始评估，主要评估以下指标：
    #1. 以pred_label选择候选个数得到的acc
    #2. 以true_label选择候选个数得到的acc
    #3. 只选择第一个作为预测结果的acc
    #4. 不同个数下的召回率
    count_with_pred_label, count_with_label, count_with_first = 0, 0, 0
    
    #5. 单个的预测准确率，多个的预测准确率, 基于pred label
    count_with_pred_one_sample, count_with_pred_multi_sample = 0, 0
    #5. 单个的预测准确率，多个的预测准确率, 基于abel
    count_with_label_one_sample, count_with_label_multi_sample = 0, 0
    recall_count_list = [0 for _ in range(len(args.hit_list))]

    count_one_sample, count_multi_sample = 0, 0

    # (raw_name, [pos_name_list], [pos_name_score], [neg_name_list], [neg_name_score], pred_count, label_count), 如果生成candidates
    candidate_list = []
    for raw_name, pos_names, sim_score_index, sim_score, pred_label, label in zip(raw_name_list, pos_name_list, similarity_matrix, similarity_score, pred_list, label_list):
        # assert len(pos_names) == label + 1
        #计算各种acc
        pred_count = []
        pred_names_list = []
        if len(pos_names) == 1:
            count_one_sample += 1
        else:
            count_multi_sample += 1
        for idx, i in enumerate([pred_label + 1, len(pos_names), 1]): #这里加1是因为label在做标签的时候时比实际长度小1的
            pred_names_in_i =[standard_name_list[x] for x in sim_score_index[:i]]
            pred_names_list.append(pred_names_in_i)
            pred_count.append(get_acc_count(pred_names_in_i, pos_names))
            if idx == 0:
                if i == 1:
                    count_with_pred_one_sample += pred_count[0]
                else:
                    count_with_pred_multi_sample += pred_count[0]
            
            if idx == 1:
                if i == 1:
                    count_with_label_one_sample += pred_count[1]
                else:
                    count_with_label_multi_sample += pred_count[1]

        count_with_pred_label += pred_count[0]
        count_with_label += pred_count[1]
        count_with_first += pred_count[2]

        #不管怎么样都算一下 正确答案在similarity中的位置 

        # (raw_name, [pos_name_list], [pos_name_score], [neg_name_list], [neg_name_score], pred_count, label_count), 如果生成candidates
        # 如果要生成candidate，则保存每个sample的一些中间结果
        # 保存所有的正例及正例的分数，以及所有前10召回的结果
        if args.generate_candidates != 'no':
            pos_names_score = [sim_score[x] for x in [standard_name_list.index(y) for y in pos_names]]
            # neg_names = [standard_name_list[x] for x in sim_score_index[:10] if standard_name_list[x] not in pos_names]
            # neg_names_score = [sim_score[x] for x in [standard_name_list.index(y) for y in neg_names]]
            cand_names = [standard_name_list[x] for x in sim_score_index[:10]]
            cand_names_score = [sim_score[x] for x in [standard_name_list.index(y) for y in cand_names]]
            candidate_list.append((raw_name, pos_names, pos_names_score, cand_names, cand_names_score, pred_label + 1, len(pos_names)))

        #开始写badcase
        if pred_count[0] == 0 and (not is_test):
            try:
                standard_name_index = [np.argwhere(sim_score_index == x) for x in [standard_name_list.index(y) for y in pos_names]][0]
            except:
                standard_name_index = [-1]
            logger.info('raw_name:{}, names_with_pred_label:{}, names_with_label:{}, standard_names:{}, standard_name_index:{}'\
                .format(raw_name, pred_names_list[0], pred_names_list[1], pos_names, standard_name_index))

        #计算recall
        for idx, hit in enumerate(args.hit_list):
            hit_names = [standard_name_list[x] for x in sim_score_index[:hit]]
            flag = True
            for pos_name in pos_names:
                if pos_name not in hit_names:
                    flag = False
                    break
            if flag:
                recall_count_list[idx] += 1

    if args.generate_candidates != 'no':
        assert len(candidate_list) != 0
        pickle.dump(candidate_list, open('./output/mto_output/{}/{}_candidate_list'.format(args.output_name, args.generate_candidates), 'wb'))
            
    acc_with_pred_label = round(count_with_pred_label / len(raw_name_list), 4)
    acc_with_pred_label_one_sample = round(count_with_pred_one_sample / count_one_sample, 4)
    acc_with_pred_label_multi_sample = round(count_with_pred_multi_sample / count_multi_sample, 4)

    acc_with_label = round(count_with_label / len(raw_name_list), 4)
    acc_with_label_one_sample = round(count_with_label_one_sample / count_one_sample, 4)
    acc_with_label_multi_sample = round(count_with_label_multi_sample / count_multi_sample, 4)
    acc_wiht_first = round(count_with_first / len(raw_name_list), 4)
    logger.info('one sample {}, multi sample {}'.format(count_one_sample, count_multi_sample))
    logger.info('pred label {}, pred label one sample {}, pred label multi sample {}'.format(count_with_pred_label, count_with_pred_one_sample, count_with_pred_multi_sample))
    logger.info('acc with pred label: {}, one sample {}, multi sample {}'.format(acc_with_pred_label, acc_with_pred_label_one_sample, acc_with_pred_label_multi_sample))
    logger.info('label {}, label one sample {}, label multi sample {}'.format(count_with_label, count_with_label_one_sample, count_with_label_multi_sample))
    logger.info('acc with label: {}, one sample {}, multi sample {}'.format(acc_with_label, acc_with_label_one_sample, acc_with_label_multi_sample))
    logger.info('acc with first label: {}'.format(acc_wiht_first))

    for hit, hit_count in zip(args.hit_list, recall_count_list):
        logger.info('hit@{}:{}'.format(hit, round(hit_count / len(raw_name_list), 4)))

    return acc_with_pred_label, acc_with_label, total_acc


if __name__ == "__main__":
    main()