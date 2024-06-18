import torch
import torch.nn as nn
# from layers.RevIN import RevIN
from transformerStruct import EnDecoder,Encoder,ConvBlock

    
class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, conv_dropout:float=0., dec_dropout:float=0.,enc_dropout:float=0.,e_layers:int=1,
                   d_layers:int=1, pe:str='zeros', n_heads:int=16, learn_pe:bool=True,
                   d_ff:int=512,d_model:int=128, au_len:int=7,imu_len:int=7,is_inference=False, 
                   c_out:int=7,attn_dropout:float=0.,pos_e:str='encoder',au_cor_matrix=None,
                   is_au_cor:bool=False,au_dropout:float=0.):
        super(Transformer, self).__init__()
        # self.revin = revin
        # if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # conv_block
        self.conv_block = ConvBlock(window=imu_len,dropout=conv_dropout,channel=106)
        self.fc2 = nn.Linear(106, c_out)  
        self.leaky_relu = nn.LeakyReLU()

        # decoder
        if pos_e == 'encoder':
            self.decoder = EnDecoder(in_channel=c_out,sq_len=au_len,d_model=d_model,d_layers=d_layers,dropout=dec_dropout,n_heads=n_heads,d_ff=d_ff,inference=is_inference)
        # elif pos_e == 'embed':
        #     self.decoder = EmbedDecoder(c_out=c_out,d_model=d_model,d_layers=d_layers,dropout=dec_dropout,n_heads=n_heads,d_ff=d_ff)
        self.projection = nn.Linear(d_model, c_out, bias=True)

        # encoder
        self.encoder = Encoder(in_channel=106, dropout=enc_dropout, sq_len=au_len, n_layers=e_layers,d_model=d_model,n_heads=n_heads,
                        d_ff=d_ff)
        # au_cor
        # self.au_cor_matrix = au_cor_matrix # au_num * au_num
        # self.is_au_cor = is_au_cor
        # self.au_fc = nn.Linear(c_out, d_model)
        self.leaky_relu = nn.LeakyReLU()
    
        # au_cor
        # self.au_cnn = AUCNN(c_out=c_out, d_model=d_model)
        # self.au_tf = AUTransformer(dropout=au_dropout,input_dim=c_out, d_model=256, nhead=n_heads,num_encoder_layers=3,dim_feedforward=1024)

    def forward(self, x, dec_in):
        features = []
        #  x:[batch, au_len60, imu_len200, channels]
        x = x.permute(0,1,3,2)
        for i in range(x.shape[1]):
            features.append(self.conv_block(x[:,i,:,:])) # [batch, imu_len, channels]
        features_stacked = torch.stack(features, dim=0) #  [au_len, batch, channels]

        # branch 
        features_bran = torch.reshape(features_stacked, (features_stacked.shape[0]* features_stacked.shape[1],features_stacked.shape[2]))

        features_bran = features_bran.view(features_bran.size(0), -1)  # 这里的-1表示自动计算展平后的特征数量
        # 通过全连接层 + 激活函数 + Dropout
        output1 = self.leaky_relu(self.fc2(features_bran))

        # decoder
        x_enc = features_stacked.permute(1,0,2) #  [ batch, au,  channels]

        # x_enc ->  [ batch, au,  106] convblocks
        # dec_in -> [ batch, au,  au_channel]

        enc_outputs, enc_self_attns = self.encoder(x_enc)
        # dec_in  [bs  x seq_len x au_channels]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_in, enc_outputs)

        output2 = self.projection(dec_outputs) # x: [bs  x seq_len x nvars]

        return output1, output2, dec_self_attns, dec_enc_attns