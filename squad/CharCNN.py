import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):

    """
    References:
    https://towardsdatascience.com/the-definitive-guide-to-bidaf-part-2-word-embedding-character-embedding-and-contextual-c151fc4f05bb
    https://arxiv.org/pdf/1408.5882.pdf
    https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/model.py

    input:
    character_matrix with shape (batch_size, sentence_length, character_matrix_height, character_matrix_width)
    number of out_channels specifying how many filters to use (int)
    kernel_height - should match element 2 of the character_matrix - (int)

    output:
    matrix with shape (character_embedding_length, N * sentence length)

    """

    def __init__(self, in_channels, out_channels, drop_prob, device):
        super(CharCNN, self).__init__()
        self.drop_prob = drop_prob

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7).to(device)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5).to(device)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3).to(device)
        self.batchnorm1 = nn.BatchNorm1d(out_channels)
        self.batchnorm2 = nn.BatchNorm1d(out_channels)
        self.batchnorm3 = nn.BatchNorm1d(out_channels)

    def forward(self, char_emb):
        batch_size, seq_len, max_word_len, embed_size = char_emb.shape
        char_emb = char_emb.transpose(2, 3).view(-1, embed_size, max_word_len)     # (batch_size * seq_len, char_embed_size, max_word_len)

        x1 = self.conv1(char_emb)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, self.drop_prob)
        x1 = self.batchnorm1(x1)
        x1_max, _ = torch.max(x1, dim=2)
        out1 = x1_max.view(batch_size, seq_len, -1)

        x2 = self.conv2(x1)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, self.drop_prob)
        x2 = self.batchnorm2(x2)
        x2_max, _ = torch.max(x2, dim=2)
        out2 = x2_max.view(batch_size, seq_len, -1)

        x3 = self.conv3(x2)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, self.drop_prob)
        x3 = self.batchnorm3(x3)
        x3_max, _ = torch.max(x3, dim=2)
        out3 = x3_max.view(batch_size, seq_len, -1)

        out = torch.cat((out1, out2, out3), 2)
        return out


class CharacterCNN(nn.Module):

    """
    References:
    https://towardsdatascience.com/the-definitive-guide-to-bidaf-part-2-word-embedding-character-embedding-and-contextual-c151fc4f05bb
    https://arxiv.org/pdf/1408.5882.pdf

    input:
    character_matrix with shape (batch_size, sentence_length, character_matrix_height, character_matrix_width)
    number of out_channels specifying how many filters to use (int)
    kernel_height - should match element 2 of the character_matrix - (int)

    output:
    matrix with shape (character_embedding_length, N * sentence length)

    """

    def __init__(self, char_cnn_out_channels, kernel_height, max_filter_width, drop_prob, device):
        super(CharacterCNN, self).__init__()
        self.device = device
        self.out_channels = char_cnn_out_channels
        self.max_filter_width = max_filter_width
        self.convs = nn.ModuleList([nn.Conv2d(1, char_cnn_out_channels, (kernel_height, w)) for w in range(1, max_filter_width + 1)])
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, character_embeddings):
        x_f = []
        for i in range(character_embeddings.shape[1]):  # for every word
            x = character_embeddings[:, i, :, :].unsqueeze(1)
            convs = [conv.to(self.device) for conv in self.convs]
            x = [F.relu(conv(x)) for conv in convs]      # (batch_size, out_channels, 1, ?)
            x_max = [F.max_pool1d(i, i.shape[3]) for i in x]      # (batch_size, out_channels, 1)
            x_max = torch.cat(x_max, 2).view(-1, self.out_channels * self.max_filter_width)
            # max_f = []
            # for conv in self.convs:
            #     conv = conv.to(self.device)
            #     x_out = F.relu(conv(x))    # (batch_size, out_channels, 1, ?)
            #     # Max-pooling
            #     x_max, _ = torch.max(x_out, dim=3)  # (batch_size, out_channels, 1)
            #     # x_max, _ = torch.max(x_max, dim=1)  # (batch_size, 1)
            #     max_f.append(x_max)
            # max_f = torch.cat(max_f, 2).view(-1, self.out_channels * self.max_filter_width)   # (batch_size, out_channels * max_filter_width)
            x_f.append(x_max)
        char_matrix = torch.stack(x_f, 1)   # (batch_size, seq_len, out_channels * 7)

        return char_matrix


if __name__ == "__main__":
    batch_size = 64
    seq_len = 387
    char_embed_size = 64
    max_word_len = 16
    out_channels = 256
    kernel_height = 7
    hidden_size = 100
    drop_prob = 0.2
    char_emb = torch.randn((batch_size, seq_len, char_embed_size, max_word_len))
    char_emb = char_emb.view(-1, char_embed_size, max_word_len)

    char_cnn = CharCNN(char_embed_size, hidden_size, drop_prob)
    char_cnn(char_emb, batch_size, seq_len)