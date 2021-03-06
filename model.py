import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
import typing
import torch.cuda
import math
import config


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        self.lstm_text = LSTMTextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )

        #self.text = TextProcessor(
        self.cnn_text = CNNTextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            #lstm_features=question_features,
            kernel_depth=question_features,
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )
        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features,
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.gate = DualGatedLinearUnit()

    def forward(self, v, q, q_len):
        q_lstm = self.lstm_text(q, list(q_len.data))
        q_cnn = self.cnn_text(q, list(q_len.data))
        q = self.gate(q_lstm, q_cnn)
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer


class DualGatedLinearUnit(nn.Module):
    def __init__(self):
        super(DualGatedLinearUnit, self).__init__()
        pass

    def forward(self, x, y):
        x_s = torch.nn.functional.sigmoid(x)
        y_s = torch.nn.functional.sigmoid(y)
        y = x_s * y + x * y_s
        return y

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


class LSTMTextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(LSTMTextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform(w)

    def forward(self, q, q_len):
        # print(q)
        embedded = self.embedding(q)
        # print(embedded.size())
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)

class ConvBlock(nn.Module):
    def __init__(self, kernel_depth, embedding_features, kernel_width, max_k=1):
        super(ConvBlock, self).__init__()
        padding_size = math.ceil((kernel_width-1)/2)

        self.cnn_conv = nn.Conv2d(1, kernel_depth, (embedding_features + 2*padding_size, kernel_width), stride=1,
                                  padding=padding_size)
        self.activation = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(kernel_depth)

    def forward(self, x):
        x = self.cnn_conv(x)
        x = self.activation(self.batchnorm(x))
        x = x.transpose(1,2)
        return x

# kernel_depth = 1024
class CNNTextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, kernel_depth, drop=0.0, kernel_width=3, layers=2, max_k=1, multilayer=True):
        super(CNNTextProcessor, self).__init__()
        self.kw = kernel_width
        self.multilayer = multilayer

        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()

        # single CNN case
        self.cnn_single_layer = ConvBlock(kernel_depth, embedding_features, kernel_width)

        # multi-layer case
        self.cnn_multi_l1 = ConvBlock(kernel_depth, embedding_features, kernel_width)
        self.cnn_multi_l2 = ConvBlock(kernel_depth, kernel_depth, kernel_width)
        self.cnn_multi_l3 = ConvBlock(kernel_depth, kernel_depth, kernel_width)

        self.pooling = nn.AdaptiveMaxPool2d((max_k, kernel_depth))

        init.xavier_uniform(self.embedding.weight)

    def forward(self, q : torch.cuda.LongTensor, q_len : int):

        embedded = self.embedding(q) # size: batch (128) * seq_len (23) * emb_len (300)
        tanhed = self.tanh(self.drop(embedded))

        c = torch.unsqueeze(tanhed.transpose(1,2), 1)  # size: batch (128) * seq_len (23) * emb_len (300)

        # single layer
        if not self.multilayer:
            c = self.cnn_single_layer(c)
        # multi layer
        else:
            c = self.cnn_multi_l1(c)
            c = self.cnn_multi_l2(c)
            c = self.cnn_multi_l3(c)

        c = self.pooling(torch.squeeze(c))
        c = torch.squeeze(c) # flatten it
        return c.squeeze(0)


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input.
        The attention map has to have the same size in all dimensions except dim=1.
    """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, c, -1)
    attention = attention.view(n, glimpses, -1)
    s = input.size(2)

    # apply a softmax to each attention map separately
    # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
    # so that each glimpse is normalized separately
    attention = attention.view(n * glimpses, -1)
    attention = F.softmax(attention)

    # apply the weighting by creating a new dim to tile both tensors over
    target_size = [n, glimpses, c, s]
    input = input.view(n, 1, c, s).expand(*target_size)
    attention = attention.view(n, glimpses, 1, s).expand(*target_size)
    weighted = input * attention
    # sum over only the spatial dimension
    weighted_mean = weighted.sum(dim=3)
    # the shape at this point is (n, glimpses, c, 1)
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled
