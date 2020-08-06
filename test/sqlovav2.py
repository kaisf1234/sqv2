import torch
from numpy import argsort
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SQLovaV2(nn.Module):
    def __init__(self):
        super(SQLovaV2, self).__init__()
        self.scp = SCP()
        self.sap = SAP()
        self.wcp = WCP()
        self.wop = WOP()
        self.wvp = WVP()

    def forward(self, context_encoding, l_q, column_encoding, l_h, g_wn):
        s_sc = self.scp(context_encoding, column_encoding, l_q, l_h)
        pr_sc = pred_sc(s_sc)
        s_sa = self.sap(context_encoding, column_encoding, l_q, l_h, pr_sc)
        s_wc = self.wcp(context_encoding, column_encoding, l_q, l_h)
        pr_wn = g_wn # For now
        pr_wc = pred_wc(pr_wn, s_wc)
        s_wo = self.wop(context_encoding, column_encoding, l_q, l_h, pr_wn, pr_wc)
        pr_wo = pred_wo(pr_wn, s_wo)
        s_wv = self.wvp(context_encoding, column_encoding, l_q, l_h, pr_wn, pr_wc, pr_wo)
        return s_sc, s_sa, None, s_wc, s_wo, s_wv   # Handle WN later


class SCP(nn.Module):
    def __init__(self):
        super(SCP, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=50,
                             num_layers=2, batch_first=True,
                             dropout=0.1, bidirectional=True)
        self.hs_map = nn.Linear(768, 100)
        hS = 100
        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.sc_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * hS, 1))

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, enc_q, enc_h, l_q, l_h):
        enc_q = enc_q[:, 1:, :, ]   # Skip CLS
        lstm_inp = pack_padded_sequence(enc_q, l_q, batch_first=True, enforce_sorted = False)
        lstm_out, _ = self.lstm(lstm_inp)
        enc_q, l_q = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        enc_h = self.hs_map(enc_h)

        ######## Taken from sqlova start ###############
        mL_n = max(l_q)

        #   [bS, mL_hs, 100] * [bS, 100, mL_n] -> [bS, mL_hs, mL_n]
        att_h = torch.bmm(enc_h, self.W_att(enc_q).transpose(1, 2))
        #   Penalty on blank parts
        for b, l_n1 in enumerate(l_q):
            if l_n1 < mL_n:
                att_h[b, :, l_n1:] = -10000000000

        p_n = self.softmax_dim2(att_h)
        #   p_n [ bS, mL_hs, mL_n]  -> [ bS, mL_hs, mL_n, 1]
        #   wenc_n [ bS, mL_n, 100] -> [ bS, 1, mL_n, 100]
        #   -> [bS, mL_hs, mL_n, 100] -> [bS, mL_hs, 100]
        c_n = torch.mul(p_n.unsqueeze(3), enc_q.unsqueeze(1)).sum(dim=2)
        vec = torch.cat([self.W_c(c_n), self.W_hs(enc_h)], dim=2)
        s_sc = self.sc_out(vec).squeeze(2)  # [bS, mL_hs, 1] -> [bS, mL_hs]
        # Penalty
        mL_hs = max(l_h)
        for b, l_hs1 in enumerate(l_h):
            if l_hs1 < mL_hs:
                s_sc[b, l_hs1:] = -10000000000
        ######## Taken from sqlova end ###############
        return s_sc

class SAP(nn.Module):
    def __init__(self):
        super(SAP, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=50,
                            num_layers=2, batch_first=True,
                            dropout=0.1, bidirectional=True)
        self.hs_map = nn.Linear(768, 100)
        hS = 100
        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.sa_out = nn.Sequential(nn.Linear(hS, hS),
                                    nn.Tanh(),
                                    nn.Linear(hS, 6))

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, enc_q, enc_h, l_q, l_h, pr_sc):
        enc_q = enc_q[:, 1:, :, ]  # Skip CLS
        lstm_inp = pack_padded_sequence(enc_q, l_q, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(lstm_inp)
        enc_q, l_q = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        enc_h = self.hs_map(enc_h)
        enc_h = enc_h[list(range(len(enc_h))), pr_sc] # Take only the correct select column and do for that
        ######## Taken from sqlova start ###############
        mL_n = max(l_q)

        att = torch.bmm(self.W_att(enc_q), enc_h.unsqueeze(2)).squeeze(2)
        for b, l_n1 in enumerate(l_q):
            if l_n1 < mL_n:
                att[b, l_n1:] = -10000000000
            # [bS, mL_n]
        p = self.softmax_dim1(att)
        c_n = torch.mul(enc_q, p.unsqueeze(2).expand_as(enc_q)).sum(dim=1)
        s_sa = self.sa_out(c_n)
        ######## Taken from sqlova end ###############
        return s_sa

class WCP(nn.Module):
    def __init__(self):
        super(WCP, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=50,
                             num_layers=2, batch_first=True,
                             dropout=0.1, bidirectional=True)
        self.hs_map = nn.Linear(768, 100)
        hS = 100
        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.sc_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * hS, 1))

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, enc_q, enc_h, l_q, l_h):
        enc_q = enc_q[:, 1:, :, ]   # Skip CLS
        lstm_inp = pack_padded_sequence(enc_q, l_q, batch_first=True, enforce_sorted = False)
        lstm_out, _ = self.lstm(lstm_inp)
        enc_q, l_q = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        enc_h = self.hs_map(enc_h)

        ######## Taken from sqlova start ###############
        mL_n = max(l_q)

        #   [bS, mL_hs, 100] * [bS, 100, mL_n] -> [bS, mL_hs, mL_n]
        att_h = torch.bmm(enc_h, self.W_att(enc_q).transpose(1, 2))
        #   Penalty on blank parts
        for b, l_n1 in enumerate(l_q):
            if l_n1 < mL_n:
                att_h[b, :, l_n1:] = -10000000000

        p_n = self.softmax_dim2(att_h)
        #   p_n [ bS, mL_hs, mL_n]  -> [ bS, mL_hs, mL_n, 1]
        #   wenc_n [ bS, mL_n, 100] -> [ bS, 1, mL_n, 100]
        #   -> [bS, mL_hs, mL_n, 100] -> [bS, mL_hs, 100]
        c_n = torch.mul(p_n.unsqueeze(3), enc_q.unsqueeze(1)).sum(dim=2)
        vec = torch.cat([self.W_c(c_n), self.W_hs(enc_h)], dim=2)
        s_wc = self.sc_out(vec).squeeze(2)  # [bS, mL_hs, 1] -> [bS, mL_hs]
        # Penalty
        mL_hs = max(l_h)
        for b, l_hs1 in enumerate(l_h):
            if l_hs1 < mL_hs:
                s_wc[b, l_hs1:] = -10000000000
        ######## Taken from sqlova end ###############
        return s_wc

class WOP(nn.Module):
    def __init__(self):
        super(WOP, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=50,
                             num_layers=2, batch_first=True,
                             dropout=0.1, bidirectional=True)
        self.hs_map = nn.Linear(768, 100)
        hS = 100
        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.mL_w = 4 # max where condition number

        self.wo_out = nn.Sequential(
            nn.Linear(2 * hS, hS),
            nn.Tanh(),
            nn.Linear(hS, 4)
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, enc_q, enc_h, l_q, l_h, wn, wc):
        enc_q = enc_q[:, 1:, :, ]   # Skip CLS
        lstm_inp = pack_padded_sequence(enc_q, l_q, batch_first=True, enforce_sorted = False)
        lstm_out, _ = self.lstm(lstm_inp)
        enc_q, l_q = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        enc_h = self.hs_map(enc_h)

        ######## Taken from sqlova start ###############
        bS = len(l_h)
        # wn

        wenc_hs_ob = []  # observed hs
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [enc_h[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [enc_h[b, 0]]  # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad)  # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob)  # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)

        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(enc_q).unsqueeze(1),
                           wenc_hs_ob.unsqueeze(3)
                           ).squeeze(3)

        # Penalty for blank part.
        mL_n = max(l_q)
        for b, l_n1 in enumerate(l_q):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000

        p = self.softmax_dim2(att)  # p( n| selected_col )

        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(enc_q.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # [bS, 5-1, dim] -> [bS, 5-1, 3]

        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob)], dim=2)
        s_wo = self.wo_out(vec)

        ######## Taken from sqlova end ###############
        return s_wo

class WVP(nn.Module):
    def __init__(self):
        super(WVP, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=50,
                             num_layers=2, batch_first=True,
                             dropout=0.1, bidirectional=True)
        self.hs_map = nn.Linear(768, 100)
        hS = 100
        self.mL_w = 4  # max where condition number

        self.W_att = nn.Linear(hS, hS)
        self.W_c = nn.Linear(hS, hS)
        self.W_hs = nn.Linear(hS, hS)
        self.W_op = nn.Linear(4, hS)
        self.n_cond_ops = 4
        self.wv_out = nn.Sequential(
            nn.Linear(4 * hS, hS),
            nn.Tanh(),
            nn.Linear(hS, 2)
        )

        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, enc_q, enc_h, l_q, l_h, wn, wc, wo):
        enc_q = enc_q[:, 1:, :, ]   # Skip CLS
        lstm_inp = pack_padded_sequence(enc_q, l_q, batch_first=True, enforce_sorted = False)
        lstm_out, _ = self.lstm(lstm_inp)
        enc_q, l_q = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        enc_h = self.hs_map(enc_h)

        ######## Taken from sqlova start ###############
        bS = len(l_h)

        wenc_hs_ob = []  # observed hs

        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            real = [enc_h[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [enc_h[b, 0]]  # this padding could be wrong. Test with zero padding later.
            wenc_hs_ob1 = torch.stack(real + pad)  # It is not used in the loss function.
            wenc_hs_ob.append(wenc_hs_ob1)

        # list to [B, 4, dim] tensor.
        wenc_hs_ob = torch.stack(wenc_hs_ob)  # list to tensor.
        wenc_hs_ob = wenc_hs_ob.to(device)

        # Column attention
        # [B, 1, mL_n, dim] * [B, 4, dim, 1]
        #  -> [B, 4, mL_n, 1] -> [B, 4, mL_n]
        # multiplication bewteen NLq-tokens and  selected column
        att = torch.matmul(self.W_att(enc_q).unsqueeze(1),
                           wenc_hs_ob.unsqueeze(3)
                           ).squeeze(3)
        # Penalty for blank part.
        mL_n = max(l_q)
        for b, l_n1 in enumerate(l_q):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000

        p = self.softmax_dim2(att)  # p( n| selected_col )


        # [B, 1, mL_n, dim] * [B, 4, mL_n, 1]
        #  --> [B, 4, mL_n, dim]
        #  --> [B, 4, dim]
        c_n = torch.mul(enc_q.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)

        # Select observed headers only.
        # Also generate one_hot vector encoding info of the operator
        # [B, 4, dim]
        wenc_op = []
        for b in range(bS):
            # [[...], [...]]
            # Pad list to maximum number of selections
            wenc_op1 = torch.zeros(self.mL_w, self.n_cond_ops)
            wo1 = wo[b]
            idx_scatter = []
            l_wo1 = len(wo1)
            for i_wo11 in range(self.mL_w):
                if i_wo11 < l_wo1:
                    wo11 = wo1[i_wo11]
                    idx_scatter.append([int(wo11)])
                else:
                    idx_scatter.append([0])  # not used anyway

            wenc_op1 = wenc_op1.scatter(1, torch.tensor(idx_scatter), 1)

            wenc_op.append(wenc_op1)

        # list to [B, 4, dim] tensor.
        wenc_op = torch.stack(wenc_op)  # list to tensor.
        wenc_op = wenc_op.to(device)

        # Now after concat, calculate logits for each token
        # [bS, 5-1, 3*hS] = [bS, 4, 300]
        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob), self.W_op(wenc_op)], dim=2)

        # Make extended vector based on encoded nl token containing column and operator information.
        # wenc_n = [bS, mL, 100]
        # vec2 = [bS, 4, mL, 400]
        vec1e = vec.unsqueeze(2).expand(-1, -1, mL_n, -1)  # [bS, 4, 1, 300]  -> [bS, 4, mL, 300]
        wenc_ne = enc_q.unsqueeze(1).expand(-1, 4, -1, -1)  # [bS, 1, mL, 100] -> [bS, 4, mL, 100]
        vec2 = torch.cat([vec1e, wenc_ne], dim=3)

        # now make logits
        s_wv = self.wv_out(vec2)  # [bS, 4, mL, 400] -> [bS, 4, mL, 2]

        # penalty for spurious tokens
        for b, l_n1 in enumerate(l_q):
            if l_n1 < mL_n:
                s_wv[b, :, l_n1:, :] = -10000000000

        ######## Taken from sqlova end ###############
        return s_wv



##### UTIL FNS FROM SQLova #########
def pred_sc(s_sc, pr_wc = None):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sc = []
    for idx, s_sc1 in enumerate(s_sc):
        if pr_wc is not None:
            s_sc1[pr_wc[idx]] = -999999
        pr_sc.append(s_sc1.argmax().item())
    return pr_sc

def pred_wc(wn, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted!
    """
    # get g_num
    pr_wc = []
    for b, wn1 in enumerate(wn):
        s_wc1 = s_wc[b]

        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())[:wn1]
        pr_wc1.sort()

        pr_wc.append(list(pr_wc1))
    return pr_wc

def pred_wo(wn, s_wo):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # s_wo = [B, 4, n_op]
    pr_wo_a = s_wo.argmax(dim=2)  # [B, 4]
    # get g_num
    pr_wo = []
    for b, pr_wo_a1 in enumerate(pr_wo_a):
        wn1 = wn[b]
        pr_wo.append(list(pr_wo_a1.data.cpu().numpy()[:wn1]))

    return pr_wo
def pred_sa(s_sa):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_sa = []
    for s_sa1 in s_sa:
        pr_sa.append(s_sa1.argmax().item())

    return pr_sa
def pred_wn(s_wn):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    # get g_num
    pr_wn = []
    for s_wn1 in s_wn:
        pr_wn.append(s_wn1.argmax().item())
        # print(pr_wn, s_wn1)
        # if s_wn1.argmax().item() == 3:
        #     input('')

    return pr_wn
def pred_wvi_se(wn, s_wv):
    """
    s_wv: [B, 4, mL, 2]
    - predict best st-idx & ed-idx
    """

    s_wv_st, s_wv_ed = s_wv.split(1, dim=3)  # [B, 4, mL, 2] -> [B, 4, mL, 1], [B, 4, mL, 1]

    s_wv_st = s_wv_st.squeeze(3) # [B, 4, mL, 1] -> [B, 4, mL]
    s_wv_ed = s_wv_ed.squeeze(3)

    pr_wvi_st_idx = s_wv_st.argmax(dim=2) # [B, 4, mL] -> [B, 4, 1]
    pr_wvi_ed_idx = s_wv_ed.argmax(dim=2)

    pr_wvi = []
    for b, wn1 in enumerate(wn):
        pr_wvi1 = []
        for i_wn in range(wn1):
            pr_wvi_st_idx11 = pr_wvi_st_idx[b][i_wn]
            pr_wvi_ed_idx11 = pr_wvi_ed_idx[b][i_wn]
            pr_wvi1.append([pr_wvi_st_idx11.item(), pr_wvi_ed_idx11.item()])
        pr_wvi.append(pr_wvi1)

    return pr_wvi
def pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_wn):
    pr_sa = pred_sa(s_sa)
    # pr_wn = pred_wn(s_wn)
    pr_wn = g_wn # For now
    pr_wc = pred_wc(pr_wn, s_wc)
    pr_sc = pred_sc(s_sc, pr_wc)
    pr_wo = pred_wo(pr_wn, s_wo)
    pr_wvi = pred_wvi_se(pr_wn, s_wv)

    return pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi


###### Loss FNs #########
def Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi):
    """

    :param s_wv: score  [ B, n_conds, T, score]
    :param g_wn: [ B ]
    :param g_wvi: [B, conds, pnt], e.g. [[[0, 6, 7, 8, 15], [0, 1, 2, 3, 4, 15]], [[0, 1, 2, 3, 16], [0, 7, 8, 9, 16]]]
    :return:
    """
    loss = 0
    loss += Loss_sc(s_sc, g_sc)
    loss += Loss_sa(s_sa, g_sa)
    #loss += Loss_wn(s_wn, g_wn)
    loss += Loss_wc(s_wc, g_wc)
    loss += Loss_wo(s_wo, g_wn, g_wo)
    loss += Loss_wv_se(s_wv, g_wn, g_wvi)

    return loss

def Loss_sc(s_sc, g_sc):
    loss = F.cross_entropy(s_sc, torch.tensor(g_sc).to(device))
    return loss

def Loss_sa(s_sa, g_sa):
    loss = F.cross_entropy(s_sa, torch.tensor(g_sa).to(device))
    return loss

def Loss_wn(s_wn, g_wn):
    loss = F.cross_entropy(s_wn, torch.tensor(g_wn).to(device))

    return loss

def Loss_wc(s_wc, g_wc):

    # Construct index matrix
    bS, max_h_len = s_wc.shape
    im = torch.zeros([bS, max_h_len]).to(device)
    for b, g_wc1 in enumerate(g_wc):
        for g_wc11 in g_wc1:
            im[b, g_wc11] = 1.0
    # Construct prob.
    p = F.sigmoid(s_wc)
    loss = F.binary_cross_entropy(p, im)

    return loss

def Loss_wo(s_wo, g_wn, g_wo):

    # Construct index matrix
    loss = 0
    for b, g_wn1 in enumerate(g_wn):
        if g_wn1 == 0:
            continue
        g_wo1 = g_wo[b]
        s_wo1 = s_wo[b]
        loss += F.cross_entropy(s_wo1[:g_wn1], torch.tensor(g_wo1).to(device))

    return loss

def Loss_wv_se(s_wv, g_wn, g_wvi):
    """
    s_wv:   [bS, 4, mL, 2], 4 stands for maximum # of condition, 2 tands for start & end logits.
    g_wvi:  [ [1, 3, 2], [4,3] ] (when B=2, wn(b=1) = 3, wn(b=2) = 2).
    """
    loss = 0
    # g_wvi = torch.tensor(g_wvi).to(device)
    for b, g_wvi1 in enumerate(g_wvi):
        # for i_wn, g_wvi11 in enumerate(g_wvi1):

        g_wn1 = g_wn[b]
        if g_wn1 == 0:
            continue
        g_wvi1 = torch.tensor(g_wvi1).to(device)
        g_st1 = g_wvi1[:,0]
        g_ed1 = g_wvi1[:,1]
        # loss from the start position
        loss += F.cross_entropy(s_wv[b,:g_wn1,:,0], g_st1)

        # print("st_login: ", s_wv[b,:g_wn1,:,0], g_st1, loss)
        # loss from the end position
        loss += F.cross_entropy(s_wv[b,:g_wn1,:,1], g_ed1)
        # print("ed_login: ", s_wv[b,:g_wn1,:,1], g_ed1, loss)

    return loss