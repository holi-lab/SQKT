import sys
sys.path.append('/home/doyounkim/sqkt/embedding')
from torch import nn

from positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """
    positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # print(f"Input to TransformerEmbedding Shape: {x.shape}")
        # Tok_emb를 사용하지 않고 바로 위치 인코딩 적용 왜냐면 이미 임베딩된 값들이기 때문
        pos_emb = self.pos_emb(x)  
        # print(f"Positional Embedding Shape: {pos_emb.shape}")
        return self.drop_out(pos_emb)