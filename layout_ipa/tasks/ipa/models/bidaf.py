import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn import LSTM, Linear


class BidafAttn(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout=0.0,
    ):
        super(BidafAttn, self).__init__()
        self.hidden_size = hidden_size

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(hidden_size, 1)
        self.att_weight_q = Linear(hidden_size, 1)
        self.att_weight_cq = Linear(hidden_size, 1)
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, c, q):
        # TODO: More memory-efficient architecture

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            # c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            # q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            # cq_tiled = c_tiled * q_tiled
            # cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                # (batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                # logger.info(f"qi {qi.shape}")
                # (batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)
            # print("C")
            # print(c.shape)
            # print("Q")
            # print(q.shape)
            # print("CQ")
            # print(cq.shape)

            # (batch, c_len, q_len)
            s = (
                self.att_weight_c(c).expand(-1, -1, q_len)
                + self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1)
                + cq
            )

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)

            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)

            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)

            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c)

            q2c_att = q2c_att.squeeze(1)
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)

            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)

            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=2)
            return x

        g = att_flow_layer(c, q)

        return g
