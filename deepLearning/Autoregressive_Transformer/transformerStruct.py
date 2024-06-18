import torch
import torch.nn as nn
from typing import Callable, Optional
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    
class PositionalEmbedding(nn.Module): # venilla
    def __init__(self, d_model, max_len=200):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)

## custom encoder

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask:Optional[Tensor]=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v = None,):
        super(MultiHeadAttention, self).__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(d_k=self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn
    
class Encoder(nn.Module):
    def __init__(self,in_channel,sq_len, dropout, n_layers,d_model,n_heads,d_ff):
        super(Encoder, self).__init__()
        self.val_emb = nn.Linear(in_channel, d_model)
        self.pos_emb = positional_encoding('zeros', learn_pe=True, q_len=sq_len, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model,n_heads,d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):

        enc_outputs = self.dropout(self.val_emb(enc_inputs) + self.pos_emb)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs,enc_self_attn_mask=None)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
    
# ----------------------
# patchTST encoder
# class TSTEncoder(nn.Module):
#     def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
#                         norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
#                         res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
#         super().__init__()

#         self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
#                                                       attn_dropout=attn_dropout, dropout=dropout,
#                                                       activation=activation, res_attention=res_attention,
#                                                       pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
#         self.res_attention = res_attention

#     def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
#         output = src
#         scores = None
#         if self.res_attention:
#             for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#             return output
#         else:
#             for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#             return output

# class TSTEncoderLayer(nn.Module):
#     def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
#                  norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
#         super().__init__()
#         assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
#         d_k = d_model // n_heads if d_k is None else d_k
#         d_v = d_model // n_heads if d_v is None else d_v

#         # Multi-Head attention
#         self.res_attention = res_attention
#         self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

#         # Add & Norm
#         self.dropout_attn = nn.Dropout(dropout)
#         if "batch" in norm.lower():
#             self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
#         else:
#             self.norm_attn = nn.LayerNorm(d_model)

#         # Position-wise Feed-Forward
#         self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
#                                 get_activation_fn(activation),
#                                 nn.Dropout(dropout),
#                                 nn.Linear(d_ff, d_model, bias=bias))

#         # Add & Norm
#         self.dropout_ffn = nn.Dropout(dropout)
#         if "batch" in norm.lower():
#             self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
#         else:
#             self.norm_ffn = nn.LayerNorm(d_model)

#         self.pre_norm = pre_norm
#         self.store_attn = store_attn


#     def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

#         # Multi-Head attention sublayer
#         if self.pre_norm:
#             src = self.norm_attn(src)
#         ## Multi-Head attention
#         if self.res_attention:
#             src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#         else:
#             src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
#         if self.store_attn:
#             self.attn = attn
#         ## Add & Norm
#         src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
#         if not self.pre_norm:
#             src = self.norm_attn(src)

#         # Feed-forward sublayer
#         if self.pre_norm:
#             src = self.norm_ffn(src)
#         ## Position-wise Feed-Forward
#         src2 = self.ff(src)
#         ## Add & Norm
#         src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
#         if not self.pre_norm:
#             src = self.norm_ffn(src)

#         if self.res_attention:
#             return src, scores
#         else:
#             return src
        
class DecoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        if enc_outputs is not None:
            dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, attn_mask=None)
            dec_outputs = self.pos_ffn(dec_outputs)
            return dec_outputs, dec_self_attn, dec_enc_attn
        else:
            dec_outputs = self.pos_ffn(dec_outputs)
            return dec_outputs, dec_self_attn, None
# decoder
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask



class EmbedDecoder(nn.Module):
    def __init__(self, c_out ,d_model,d_layers,dropout,n_heads,d_ff):
        super(EmbedDecoder, self).__init__()
        self.val_emb = nn.Linear(c_out, d_model)
        self.pos_emb = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        # self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(de_in_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(d_model,n_heads, d_ff) for _ in range(d_layers)])

    def forward(self, dec_inputs, enc_outputs): # dec_inputs : [batch_size x target_len x channels]
        # dec_re = dec_inputs.reshape((dec_inputs.shape[0], dec_inputs.shape[1], 1))
        dec_outputs = self.dropout(self.val_emb(dec_inputs) + self.pos_emb(dec_inputs))
        device = dec_inputs.device
        # dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(device)
        # dec_self_attn_mask = torch.gt(dec_self_attn_subsequent_mask, 0)

        # dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask=None)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns



class EnDecoder(nn.Module):
    def __init__(self, in_channel,sq_len,d_model,d_layers,dropout,n_heads,d_ff,inference):
        super(EnDecoder, self).__init__()
        self.val_emb = nn.Linear(in_channel, d_model)
        self.pos_emb = positional_encoding('zeros', learn_pe=True, q_len=sq_len, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        # self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(de_in_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(d_model,n_heads, d_ff) for _ in range(d_layers)])
        self.inference = inference
    def forward(self, dec_inputs, enc_outputs): # dec_inputs : [batch_size x target_len x channels]
        # dec_re = dec_inputs.reshape((dec_inputs.shape[0], dec_inputs.shape[1], 1))
        if self.inference == True:
            pos_emd = self.pos_emb[:dec_inputs.shape[1],:]
        else:
            pos_emd = self.pos_emb
        dec_outputs = self.dropout(self.val_emb(dec_inputs) + pos_emd)
        device = dec_inputs.device
        # dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).to(device)
        dec_self_attn_mask = torch.gt(dec_self_attn_subsequent_mask, 0)

        # dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class ConvBlock(nn.Module):
    def __init__(self, window,dropout,channel):
        super(ConvBlock, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=48, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=48, out_channels=96, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(in_channels=96, out_channels=108, kernel_size=3, stride=1)
        # 最大池化层
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        # LeakyReLU激活函数
        self.leaky_relu = nn.LeakyReLU()
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.patch = int(((window-2*3)/2-2)//2)
        self.fc1 = nn.Linear(108 * self.patch, channel) 

    def forward(self, x):
        # 通过卷积层 + 激活函数 + 池化层
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.leaky_relu(self.conv4(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # 这里的-1表示自动计算展平后的特征数量
        # 通过全连接层 + 激活函数 + Dropout
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        return x
    
class AUTransformer(nn.Module):
    def __init__(self, dropout, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(AUTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model) # input_dim * input_dim
        self.dropout = nn.Dropout(p=dropout)
        self.pos_emb = positional_encoding('zeros', learn_pe=True, q_len=input_dim, d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src):
        # 假设src的形状是(batch_size, seq_len, input_dim)，这里seq_len和input_dim都是13
        src = self.dropout(self.embedding(src) + self.pos_emb) # 维度变为(batch_size, seq_len, d_model)
        output = self.transformer_encoder(src)
        output = self.output_layer(output)  # 维度变为(batch_size, seq_len, 128)
        output = output.squeeze(-1)
        return output
    

class AUCNN(nn.Module):
    def __init__(self,c_out, d_model):
        super(AUCNN, self).__init__()
        # 假设输入是 (batch_size, 1, 13, 13)，其中1是通道数
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(7*7*64, c_out*d_model)  # 通过计算得到的尺寸
        self.leaky_relu = nn.LeakyReLU()
        self.c_out = c_out
        self.d_model = d_model

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = x.view(-1, 7*7*64)  # 展平
        x = self.fc(x)
        x = x.view(-1, self.c_out, self.d_model)  # 调整输出形状为 (batch_size, 13, 128)
        return x