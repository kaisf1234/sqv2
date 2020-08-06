
######### Imports ###########
import json
import os
import sys
import time

import torch
from data_utils import read_data
from datasets import TaBertDataset
from batch_processors import pre_proc
from old_sqlova import get_g, convert_pr_wvi_to_string, sort_pr_wc, generate_sql_i, get_cnt_sw_list, get_cnt_lx_list
from old_sqlova import get_g_wvi_corenlp, get_t_tt_indexes, get_g_wvi_bert_from_g_wvi_corenlp
from pytorch_pretrained_bert import BertTokenizer
from sqlovav2 import SQLovaV2, Loss_sw_se, pred_sw_se
from tabert_utils import tabert_tokenize_tables, tabert_get_contexts
from torch import nn
from torch.utils.data import DataLoader
from table_bert import Table, Column
from table_bert import TableBertModel

sys.path.insert(1, "./")
######### Important consts ###########
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
batch_size = 4
accumulate_gradients = 8

model_store_root = ""
run_key = ""
data_root = "./data_root/clean"
column_vector_path = "./column_vec/all"
lr = 1e-3
lr_bert = 1.5e-6
######### Datasets ###########

train_data, train_tables = read_data(data_root, "train")
dev_data, dev_tables = read_data(data_root, "dev")


train_set = TaBertDataset(train_data)
dev_set = TaBertDataset(dev_data)
train_loader = DataLoader(train_set, batch_size= batch_size, shuffle=False, num_workers=16 if cuda else 0, collate_fn= lambda x:x)
dev_loader = DataLoader(dev_set, batch_size= batch_size, shuffle=False, num_workers=16 if cuda else 0, collate_fn=lambda x:x)


with open(column_vector_path + "/dev_vecs.json") as f:
    dev_column_vectors = json.load(f)

with open(column_vector_path + "/train_vecs.json") as f:
    train_column_vectors = json.load(f)





######### Models / Tokenizer / Optimizers ###########

model_bert = TableBertModel.from_pretrained(
    './tabert_base_k3/model.bin',
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = SQLovaV2() # Dummy for now

model_bert.to(device)
model.to(device)
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr)

opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                             lr=lr_bert)


######### Loading Models Optimizers ###########

#TODO: Figure this out

######### Train / Test Loop ###########

def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )



def train():
    model.train()
    model_bert.train()
    ave_loss = 0
    cnt = 0  # count the # of examples
    cnt_sc = 0  # count the # of correct predictions of select column
    cnt_sa = 0  # of selectd aggregation
    cnt_wn = 0  # of where number
    cnt_wc = 0  # of where column
    cnt_wo = 0  # of where operator
    cnt_wv = 0  # of where-value
    cnt_wvi = 0  # of where-value index (on question tokens)
    cnt_lx = 0  # of logical form acc
    cnt_x = 0  # of execution acc
    start = time.time()
    l = len(train_loader)
    for iB, batch in enumerate(train_loader):
        if iB % 2 == 1:
            print("Done with ", iB, " out of ", l, " time left ", ((time.time() - start) / iB) * (l - iB), " ave loss ",
                  ave_loss / cnt)

        nlus, sqls, headers, values, nlu_origs = pre_proc(batch, train_tables, train_column_vectors)
        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sqls)
        g_wvi_corenlp = get_g_wvi_corenlp(batch)
        t_to_tt_idx, tt_to_t_idx, l_q, nlu_tts = get_t_tt_indexes(nlus, tokenizer) # Just need index between words and word-pieces
        g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)

        tabert_tokenized_tables = tabert_tokenize_tables(headers, values, [query["table_id"] for query in batch], tokenizer)
        tabert_contexts = tabert_get_contexts(nlus, tokenizer)
        context_encoding, column_encoding, info_dict = model_bert.encode(
            contexts=tabert_contexts,
            tables=tabert_tokenized_tables
        )
        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model.forward(context_encoding, l_q, column_encoding, [len(x) for x in headers], g_wn)
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

        if iB % accumulate_gradients == 0:  # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
                    # scheduler.step()
        elif iB % accumulate_gradients == (accumulate_gradients - 1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
                # scheduler.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_wn)
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlus, nlu_tts, tt_to_t_idx)

        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu_origs)

        # Cacluate accuracy
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                      pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                      sqls, pr_sql_i,
                                                      mode='train')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        # lx stands for logical form accuracy
        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = cnt_lx1_list, [], [],  # get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # statistics
        ave_loss += loss.item()

        # count
        cnt += len(nlus)
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]

    aux_out = 1

    return acc, aux_out

for epoch in range(100):
    acc_train, _ = train()
    print_result(epoch, acc_train, 'train')
    state = {'model': model.state_dict()}
    torch.save(state, os.path.join(model_store_root, run_key + str(epoch) + 'model_best_orig.pt'))

    state = {'model_bert': model_bert.state_dict()}
    torch.save(state, os.path.join(model_store_root, run_key + str(epoch) + 'model_bert_best_orig.pt'))

    torch.save(opt.state_dict(),
               os.path.join(model_store_root, run_key + str(epoch) + 'opt_best_orig.pt'))
    torch.save(opt_bert.state_dict(),
               os.path.join(model_store_root, run_key + str(epoch) + 'opt_bert_best_orig.pt'))
