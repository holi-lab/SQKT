from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        # super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
        super(TokenEmbedding, self).__init__(vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x의 입력 차원을 확인하고, 필요하면 조정
        print(f"Input to TokenEmbedding Shape: {x.shape}")
        x = self.embedding(x)
        print(f"Output from TokenEmbedding Shape: {x.shape}")
        return x