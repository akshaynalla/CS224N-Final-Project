from torch import nn
import torch
from util import masked_softmax


class AnswerPointer(nn.Module):
    """
    References:
    https://arxiv.org/pdf/1608.07905.pdf
    https://github.com/laddie132/Match-LSTM/blob/793f0ecd1b1deb69a690602d505cdffe35696be6/models/layers.py
    https://github.com/geraltofrivia/match-lstm-ptr-network/blob/master/networks.py
    https://github.com/xingdi-eric-yuan/MatchLSTM-PyTorch/blob/5471169c5796b7a52542a1ff41f075915b2fbde0/lib/layers/eric_temp_layers.py

    """
    def __init__(self, input_size, hidden_size, device):
        super(AnswerPointer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # H_r: (P, 2 * hidden_size)
        # V: (2 * hidden_size, hidden_size)
        # W_a: (hidden_size, hidden_size)
        # h_ak: (hidden_size)
        # b_a: (hidden_size)
        # v: (hidden_size)

        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(input_size, hidden_size, bias=False)
        # self.v = nn.Parameter(torch.randn(hidden_size, 1))
        self.v = nn.Linear(hidden_size, 1)
        # self.c = nn.Parameter(torch.randn(1))
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.W_a.weight.data)
        torch.nn.init.xavier_uniform_(self.V.weight.data)
        torch.nn.init.xavier_uniform_(self.v.weight.data)
        # torch.nn.init.xavier_uniform_(self.c)

    def forward(self, H_r, mask):
        batch_size, P, _ = H_r.shape
        betas = []

        h_ak = torch.zeros((batch_size, self.hidden_size))
        c_ak = torch.zeros((batch_size, self.hidden_size))

        h_ak = h_ak.to(self.device)
        c_ak = c_ak.to(self.device)

        for _ in range(2):
            # F_k: (batch_size, P, hidden_size)
            F_k = torch.tanh(self.V(H_r) + self.W_a(h_ak).unsqueeze(1).repeat(1, P, 1))

            # F_k * v: (batch_size, P, 1)
            # beta_k: (batch_size, P)
            # beta_k = masked_softmax(self.v(F_k) + self.c.repeat(batch_size, P, 1), mask)
            beta_k = masked_softmax(self.v(F_k).squeeze(), mask)
            betas.append(masked_softmax(self.v(F_k).squeeze(), mask, log_softmax=True))
            # rnn_input: (batch_size, 2 * hidden_size, 1)
            rnn_input = torch.bmm(H_r.transpose(1, 2), beta_k.unsqueeze(2)).squeeze()
            h_ak, c_ak = self.rnn(rnn_input, (h_ak, c_ak))

        # betas = torch.stack(betas, dim=0)
        return betas


    #     self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
    #     self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)
    #
    #     self.rnn = RNNEncoder(input_size=2 * hidden_size,
    #                           hidden_size=hidden_size,
    #                           num_layers=1,
    #                           drop_prob=drop_prob)
    #
    #     self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
    #     self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)
    #
    # def forward(self, att, mod, mask):
    #     # Shapes: (batch_size, seq_len, 1)
    #     logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
    #     mod_2 = self.rnn(mod, mask.sum(-1))
    #     logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)
    #
    #     # Shapes: (batch_size, seq_len)
    #     log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
    #     log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
    #
    #     return log_p1, log_p2


if __name__ == "__main__":
    batch_size = 32
    P = 284
    hidden_size = 64
    mod = torch.randn((batch_size, P, 2 * hidden_size))

    dec = AnswerPointer(2*hidden_size, hidden_size)
    h_ak = torch.randn((batch_size, hidden_size))
    c_mask = torch.zeros((batch_size, 284))
    betas = dec(mod, c_mask)
