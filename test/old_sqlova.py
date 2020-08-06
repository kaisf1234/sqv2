from copy import deepcopy
import numpy as np

def get_wc1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wc1 = []
    for cond in conds:
        wc1.append(cond[0])
    return wc1


def get_wo1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wo1 = []
    for cond in conds:
        wo1.append(cond[1])
    return wo1


def get_wv1(conds):
    """
    [ [wc, wo, wv],
      [wc, wo, wv], ...
    ]
    """
    wv1 = []
    for cond in conds:
        wv1.append(cond[2])
    return wv1


def get_g(sql_i):
    """ for backward compatibility, separated with get_g"""
    g_sc = []
    g_sa = []
    g_wn = []
    g_wc = []
    g_wo = []
    g_wv = []
    for b, psql_i1 in enumerate(sql_i):
        g_sc.append( psql_i1["sel"] )
        g_sa.append( psql_i1["agg"])

        conds = psql_i1['conds']
        if not psql_i1["agg"] < 0:
            g_wn.append( len( conds ) )
            g_wc.append( get_wc1(conds) )
            g_wo.append( get_wo1(conds) )
            g_wv.append( get_wv1(conds) )
        else:
            raise EnvironmentError
    return g_sc, g_sa, g_wn, g_wc, g_wo, g_wv

def get_g_wvi_corenlp(t):
    g_wvi_corenlp = []
    for t1 in t:
        g_wvi_corenlp.append( t1['wvi_corenlp'] )
    return g_wvi_corenlp

def get_t_tt_indexes(queries, tokenizer):
    t_to_tt_idx = []
    tt_to_t_idx = []
    l_n = []
    nlu_tts = []
    for _, query in enumerate(queries):
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (i, token) in enumerate(query):
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(i)
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer
        tt_to_t_idx.append(tt_to_t_idx1)
        t_to_tt_idx.append(t_to_tt_idx1)
        nlu_tts.append(nlu_tt1)
        l_n.append(len(nlu_tt1))
    return t_to_tt_idx, tt_to_t_idx, l_n, nlu_tts


def get_g_wvi_bert_from_g_wvi_corenlp(wh_to_wp_index, g_wvi_corenlp):
    """
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.

    Assumption: where_str always presents in the nlu.
    """
    g_wvi = []
    for b, g_wvi_corenlp1 in enumerate(g_wvi_corenlp):
        wh_to_wp_index1 = wh_to_wp_index[b]
        g_wvi1 = []
        for i_wn, g_wvi_corenlp11 in enumerate(g_wvi_corenlp1):

            st_idx, ed_idx = g_wvi_corenlp11

            st_wp_idx = wh_to_wp_index1[st_idx]
            ed_wp_idx = wh_to_wp_index1[ed_idx]

            g_wvi11 = [st_wp_idx, ed_wp_idx]
            g_wvi1.append(g_wvi11)

        g_wvi.append(g_wvi1)

    return g_wvi



def convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_wp_t, wp_to_wh_index):
    """
    - Convert to the string in whilte-space-separated tokens
    - Add-hoc addition.
    """
    pr_wv_str_wp = [] # word-piece version
    pr_wv_str = []
    for b, pr_wvi1 in enumerate(pr_wvi):
        pr_wv_str_wp1 = []
        pr_wv_str1 = []
        wp_to_wh_index1 = wp_to_wh_index[b]
        nlu_wp_t1 = nlu_wp_t[b]
        nlu_t1 = nlu_t[b]

        for i_wn, pr_wvi11 in enumerate(pr_wvi1):
            st_idx, ed_idx = pr_wvi11

            # Ad-hoc modification of ed_idx to deal with wp-tokenization effect.
            # e.g.) to convert "butler cc (" ->"butler cc (ks)" (dev set 1st question).
            pr_wv_str_wp11 = nlu_wp_t1[st_idx:ed_idx+1]
            pr_wv_str_wp1.append(pr_wv_str_wp11)

            st_wh_idx = wp_to_wh_index1[st_idx]
            ed_wh_idx = wp_to_wh_index1[ed_idx]
            pr_wv_str11 = nlu_t1[st_wh_idx:ed_wh_idx+1]

            pr_wv_str1.append(pr_wv_str11)

        pr_wv_str_wp.append(pr_wv_str_wp1)
        pr_wv_str.append(pr_wv_str1)

    return pr_wv_str, pr_wv_str_wp

def sort_pr_wc(pr_wc, g_wc):
    """
    Input: list
    pr_wc = [B, n_conds]
    g_wc = [B, n_conds]


    Return: list
    pr_wc_sorted = [B, n_conds]
    """
    pr_wc_sorted = []
    for b, pr_wc1 in enumerate(pr_wc):
        g_wc1 = g_wc[b]
        pr_wc1_sorted = []

        if set(g_wc1) == set(pr_wc1):
            pr_wc1_sorted = deepcopy(g_wc1)
        else:
            # no sorting when g_wc1 and pr_wc1 are different.
            pr_wc1_sorted = deepcopy(pr_wc1)

        pr_wc_sorted.append(pr_wc1_sorted)
    return pr_wc_sorted

def generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu):
    pr_sql_i = []
    for b, nlu1 in enumerate(nlu):
        conds = []
        for i_wn in range(pr_wn[b]):
            conds1 = []
            conds1.append(pr_wc[b][i_wn])
            conds1.append(pr_wo[b][i_wn])
            merged_wv11 = merge_wv_t1_eng(pr_wv_str[b][i_wn], nlu[b])
            conds1.append(merged_wv11)
            conds.append(conds1)

        pr_sql_i1 = {'agg': pr_sa[b], 'sel': pr_sc[b], 'conds': conds}
        pr_sql_i.append(pr_sql_i1)
    return pr_sql_i

def merge_wv_t1_eng(where_str_tokens, NLq):
    """
    Almost copied of SQLNet.
    The main purpose is pad blank line while combining tokens.
    """
    nlq = NLq.lower()
    where_str_tokens = [tok.lower() for tok in where_str_tokens]
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$'
    special = {'-LRB-': '(',
               '-RRB-': ')',
               '-LSB-': '[',
               '-RSB-': ']',
               '``': '"',
               '\'\'': '"',
               }
               # '--': '\u2013'} # this generate error for test 5661 case.
    ret = ''
    double_quote_appear = 0
    for raw_w_token in where_str_tokens:
        # if '' (empty string) of None, continue
        if not raw_w_token:
            continue

        # Change the special characters
        w_token = special.get(raw_w_token, raw_w_token)  # maybe necessary for some case?

        # check the double quote
        if w_token == '"':
            double_quote_appear = 1 - double_quote_appear

        # Check whether ret is empty. ret is selected where condition.
        if len(ret) == 0:
            pass
        # Check blank character.
        elif len(ret) > 0 and ret + ' ' + w_token in nlq:
            # Pad ' ' if ret + ' ' is part of nlq.
            ret = ret + ' '

        elif len(ret) > 0 and ret + w_token in nlq:
            pass  # already in good form. Later, ret + w_token will performed.

        # Below for unnatural question I guess. Is it likely to appear?
        elif w_token == '"':
            if double_quote_appear:
                ret = ret + ' '  # pad blank line between next token when " because in this case, it is of closing apperas
                # for the case of opening, no blank line.

        elif w_token[0] not in alphabet:
            pass  # non alphabet one does not pad blank line.

        # when previous character is the special case.
        elif (ret[-1] not in ['(', '/', '\u2013', '#', '$', '&']) and (ret[-1] != '"' or not double_quote_appear):
            ret = ret + ' '
        ret = ret + w_token

    return ret.strip()

def get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                    pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                    g_sql_i, pr_sql_i,
                    mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_sc = get_cnt_sc_list(g_sc, pr_sc)
    cnt_sa = get_cnt_sc_list(g_sa, pr_sa)
    cnt_wn = get_cnt_sc_list(g_wn, pr_wn)
    cnt_wc = get_cnt_wc_list(g_wc, pr_wc)
    cnt_wo = get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode)
    if pr_wvi:
        cnt_wvi = get_cnt_wvi_list(g_wn, g_wc, g_wvi, pr_wvi, mode)
    else:
        cnt_wvi = [0]*len(cnt_sc)
    cnt_wv = get_cnt_wv_list(g_wn, g_wc, g_sql_i, pr_sql_i, mode) # compare using wv-str which presented in original data.


    return cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wvi, cnt_wv

def get_cnt_sc_list(g_sc, pr_sc):
    cnt_list = []
    for b, g_sc1 in enumerate(g_sc):
        pr_sc1 = pr_sc[b]
        if pr_sc1 == g_sc1:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list

def get_cnt_wc_list(g_wc, pr_wc):
    cnt_list= []
    for b, g_wc1 in enumerate(g_wc):

        pr_wc1 = pr_wc[b]
        pr_wn1 = len(pr_wc1)
        g_wn1 = len(g_wc1)

        if pr_wn1 != g_wn1:
            cnt_list.append(0)
            continue
        else:
            wc1 = np.array(g_wc1)
            wc1.sort()

            if np.array_equal(pr_wc1, wc1):
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list

def get_cnt_wo_list(g_wn, g_wc, g_wo, pr_wc, pr_wo, mode):
    """ pr's are all sorted as pr_wc are sorted in increasing order (in column idx)
        However, g's are not sorted.

        Sort g's in increasing order (in column idx)
    """
    cnt_list=[]
    for b, g_wo1 in enumerate(g_wo):
        g_wc1 = g_wc[b]
        pr_wc1 = pr_wc[b]
        pr_wo1 = pr_wo[b]
        pr_wn1 = len(pr_wo1)
        g_wn1 = g_wn[b]

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            # Sort based wc sequence.
            if mode == 'test':
                idx = np.argsort(np.array(g_wc1))

                g_wo1_s = np.array(g_wo1)[idx]
                g_wo1_s = list(g_wo1_s)
            elif mode == 'train':
                # due to tearch forcing, no need to sort.
                g_wo1_s = g_wo1
            else:
                raise ValueError

            if type(pr_wo1) != list:
                raise TypeError
            if g_wo1_s == pr_wo1:
                cnt_list.append(1)
            else:
                cnt_list.append(0)
    return cnt_list

def get_cnt_wvi_list(g_wn, g_wc, g_wvi, pr_wvi, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_list =[]
    for b, g_wvi1 in enumerate(g_wvi):
        g_wc1 = g_wc[b]
        pr_wvi1 = pr_wvi[b]
        pr_wn1 = len(pr_wvi1)
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = np.argsort(np.array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi11 = g_wvi1[idx11]
                pr_wvi11 = pr_wvi1[i_wn]
                if g_wvi11 != pr_wvi11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list

def get_cnt_wv_list(g_wn, g_wc, g_sql_i, pr_sql_i, mode):
    """ usalbe only when g_wc was used to find pr_wv
    """
    cnt_list =[]
    for b, g_wc1 in enumerate(g_wc):
        pr_wn1 = len(pr_sql_i[b]["conds"])
        g_wn1 = g_wn[b]

        # Now sorting.
        # Sort based wc sequence.
        if mode == 'test':
            idx1 = np.argsort(np.array(g_wc1))
        elif mode == 'train':
            idx1 = list( range( g_wn1) )
        else:
            raise ValueError

        if g_wn1 != pr_wn1:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g_wvi_str11 = str(g_sql_i[b]["conds"][idx11][2]).lower()
                pr_wvi_str11 = str(pr_sql_i[b]["conds"][i_wn][2]).lower()
                # print(g_wvi_str11)
                # print(pr_wvi_str11)
                # print(g_wvi_str11==pr_wvi_str11)
                if g_wvi_str11 != pr_wvi_str11:
                    flag = False
                    # print(g_wv1, g_wv11)
                    # print(pr_wv1, pr_wv11)
                    # input('')
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)

    return cnt_list

def get_cnt_lx_list(cnt_sc1, cnt_sa1, cnt_wn1, cnt_wc1, cnt_wo1, cnt_wv1):
    # all cnt are list here.
    cnt_list = []
    cnt_lx = 0
    for csc, csa, cwn, cwc, cwo, cwv in zip(cnt_sc1, cnt_sa1, cnt_wn1, cnt_wc1, cnt_wo1, cnt_wv1):
        if csc and csa and cwn and cwc and cwo and cwv:
            cnt_list.append(1)
        else:
            cnt_list.append(0)

    return cnt_list