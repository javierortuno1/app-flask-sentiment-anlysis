import torch.nn as nn


class SentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        # EmbeddingBag is the combination of Embedding and mean() in a single layer
        # TODO complete the embedding bag and fc layers with the correct parameters. Set `sparse`=True in the EmbeddingBag
        # EmbeddingBag is the combination of Embedding and mean() in a single layer
        # EmbeddingBag averages all embeddings
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        # TODO complete the forward method. EmbeddingBag layers take `text` and `offsets` as inputs
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
