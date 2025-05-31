
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on
    Key : every sentence to check relationship with Query
    Value : every sentence same with Key
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask = None, e=1e-12):
        # input is 4 dimension tensor
        """
        input:
            'query' : [batch_size, head, length, d_tensor(d_q)]
            'key' : [batch_size, head, length, d_tensor(d_k)]
            'value' : [batch_size, head, length, d_tensor(d_v)]
            'dropoutp' : nn.Dropout

        return:
            'weighted value' : [batch_size, head, length, d_tensor(d_v)]
            'weight matrix' : [batch_size, head, length, length]
        """

        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = torch.matmul(q, k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. Apply mask (if provided) before softmax
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))  # 마스크된 부분을 -inf로 설정하여 무시


        # 2. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 3. multiply with Value
        v = torch.matmul(score, v)

        return v