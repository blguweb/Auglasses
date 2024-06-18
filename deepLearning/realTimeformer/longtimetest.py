import torch
import pickle
import torch.nn as nn  
import argparse
from modelStruct import Convformer
import numpy as np
import pandas as pd
from data_provider.dataLoader import My_Dataset
import os
from metrics import metric
from utils import *
from sklearn import svm
from torch.utils.data import Dataset

def visual_pred_lens(pred_lens, output_channels, true=None, preds=None,  name='./pic/test.pdf'):
    """
    Results visualization
    """
    rows = 4  # 选择适当的行数
    cols = output_channels // rows + (output_channels % rows > 0)

    fig, axs = plt.subplots(rows, cols, figsize=(20, 10))  # 调整大小以适应所有子图
    axs = axs.flatten()  # 将多维数组展平，便于索引
    x = np.arange(0, pred_lens)
    for i in range(output_channels):
        axs[i].plot(x, preds[:, i], label='Prediction', linewidth=2)  # 绘制模型输出
        axs[i].plot(x, true[:, i], label='GroundTruth', linewidth=2)  # 绘制实际值
        axs[i].set_title(f'Channel {i+1}')  # 设置子图标题
        axs[i].legend()  # 显示图例
        axs[i].set_ylim(0, 5)  # 设置纵坐标范围

    # 对于不需要的子图位置，关闭它们
    for i in range(output_channels, rows*cols):
        fig.delaxes(axs[i])
    plt.legend()
    plt.tight_layout()  # 调整子图间距
    plt.savefig(name, bbox_inches='tight')
    plt.close()

class IMU_Dataset(Dataset):
    """自定义数据集"""

    def __init__(self, data_x, data_y):
        self.x = data_x
        self.y = data_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        label = np.delete(self.y[item],(9,11,16), axis=1)
        return self.x[item], label
def channel_mae(y_pred, y_true):
    # 计算每个样本的每个通道的绝对误差
    absolute_errors = np.abs(y_true - y_pred)
    # 计算每个通道的平均绝对误差
    channel_maes = np.mean(absolute_errors, axis=(0, 1))
    mse_errors = (y_pred - y_true) ** 2
    channel_mses = np.mean(mse_errors, axis=(0, 1))
    return channel_maes, channel_mses

    
def feature_caculate(au_array):
    # [expression_len, au_channels]
    feature = []
    feature = np.nanmean(au_array,axis=0).tolist()
    feature.extend(np.nanvar(au_array,axis=0).tolist())
    feature.extend(fIAV(au_array))
    feature.extend(fMAX(au_array))
    feature.extend(fRMS(au_array))
    return feature
  
def norm(data_x):
    for i in range(data_x.shape[1]):
        mean = np.mean(data_x[:,i])
        std = np.std(data_x[:,i])
        data_x[:,i] = [(num-mean)/std for num in data_x[:,i]]
    # # scaled之后的数据零均值，单位方差
    print(data_x.mean(axis=0)[:5])  # column mean: array([ 0.,  0.,  0.])
    print(data_x.std(axis=0)[:5])  # column standard deviation: array([ 1.,  1.,  1.])
    return data_x

def initModel(opt,device):
    criterion = nn.MSELoss()
    weights_dir = os.path.join('./log/', opt.time_run, opt.user, opt.model_name,"weights")
    weight_path = weights_dir + '/checkpoint.pth'
    assert os.path.exists(weight_path), "model path: {} does not exists".format(weight_path)
    
    model = Convformer(conv_dropout=args.conv_dropout,
                        dec_dropout=args.tf_dropout,
                        enc_dropout=args.tf_dropout,
                        n_heads=args.n_heads,
                        e_layers=args.e_layers,
                        d_layers=args.d_layers,
                        d_ff=args.d_ff,
                        d_model=args.d_model,
                        au_len=args.au_len,
                        imu_len=args.imu_len,
                        c_out=args.c_out,
                        pos_e=args.pos_e,
                        au_cor_matrix=None,
                        is_au_cor=args.au_cor)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    model.eval()
    return criterion, model


def model_test(opt, imu_data, au_data, device, criterion, model):
    # batch size = 1
    imu_data = imu_data[np.newaxis, :]
    au_data = au_data[np.newaxis, :]
    # cut
    test_dataset = IMU_Dataset(imu_data,au_data)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                    num_workers=1, drop_last=False)
    
    test_path = os.path.join('./longTimeResult/', opt.time_run, opt.downstream_dataset, opt.model_name, str(opt.total_length))
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    labels = []
    outputs = []
    result_loss = []
    total_loss = []
    total_mae =[]
    pd = None
    # 测试模型（这里简单使用训练数据进行演示，实际应使用独立的测试集）
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            # batch_x = batch_x.float().to(device)
            # batch_y = batch_y.float().to(device)

            # init
            for times in range(0, batch_y.shape[1],opt.total_length):
                if (times+opt.total_length) > batch_y.shape[1]:
                    continue
                result = batch_y[:,times:times+opt.au_len,:].float().to(device)

                
                for se_i in range(0, opt.total_length, opt.pred_len):
                    se_i = times + se_i # 绝对位置

                    if (se_i+opt.au_len) >= (opt.total_length + times):
                        continue
                    intput_x = batch_x[:,se_i:se_i+opt.au_len,:].float().to(device)
                    

                    # intput_y = batch_y[:,se_i:se_i+opt.au_len,:].float().to(device)

                    dec_inp = torch.zeros_like(result[:, -opt.pred_len:, :]).float()
                    dec_inp = torch.cat([result[:, se_i-times:(se_i + opt.au_len - opt.pred_len -times), :], dec_inp], dim=1).float().to(device)
                    # print(intput_x.shape, dec_inp.shape)

                    output_conv, output_former, dec_self_attns, dec_enc_attns = model(intput_x, dec_inp)
                    # output_former: 
                    mi_result = result.cpu().numpy().squeeze(0)
                    mi_former = output_former.cpu().numpy().squeeze(0)
                    # plot_compare(mi_result,mi_former,mi_result.shape[0] + opt.pred_len, opt.pred_len,se_i,emotion)
                    
                    if se_i == 0 + times:
                        result = output_former
                    else:
                        result = torch.cat([result, output_former[:,-opt.pred_len:, :]], dim=1) # 只用预测部分


                print('inference finished!')
                gt = batch_y[:,times:times+opt.total_length,:].cpu().numpy().squeeze(0)
                pd = result.cpu().numpy().squeeze(0)
                mae, mse, rmse, rse, corr = metric(pd, gt[:pd.shape[0], :])
                total_mae.append(mae)
                print(times)
                visual_pred_lens(pd.shape[0], opt.c_out, true=gt[:pd.shape[0], :], preds=pd, name=os.path.join(test_path, str(times) + '.pdf'))
    # result_loss = np.average(result_loss)
    # total_loss = np.average(total_loss)
    # outputs = torch.stack(outputs, dim=0).numpy()
    # labels = torch.stack(labels, dim=0).numpy()
    # outputs = outputs.reshape(-1, outputs.shape[-2], outputs.shape[-1])
    # labels = labels.reshape(-1, labels.shape[-2], labels.shape[-1])
    # ## 因为变成1
    # outputs = outputs.squeeze(0)
    # labels = labels.squeeze(0)
    # mae, mse, rmse, rse, corr = metric(outputs, labels)
    # print('Test: total_loss :{}, result loss: {}, mse:{}, mae:{}, rse:{}'.format(total_loss, result_loss,mse, mae, rse))
    print(f'total_mae: {total_mae}')
    print('mean mae: ' ,np.mean(total_mae))
    # return pd, gt[:pd.shape[0], :]

def imuInference(opt, imu, au, device, window_length):
    # [imu/au_len, time + channels]
    # make to : au_len , [au_len , imu_len, channels]

    criterion, model = initModel(opt,device)
    imu_dataset = []
    au_gt = []

    for au_index in range(au.shape[0]):
        imu_m_index = find_index(au[au_index, 0], imu[:, 0])
        if imu_m_index == None:
            continue
        imu_start_index = imu_m_index - window_length//2
        imu_end_index = imu_start_index + window_length
        if imu_end_index >= imu.shape[0] or imu_start_index < 0:
            continue

        # Selecting the imu data in the window
        imu_dataset.append(imu[imu_start_index:imu_end_index, 1:])
        au_gt.append(au[au_index, 1:])
    # 
    imu_dataset = np.array(imu_dataset) # [expression_len, window_length, imu_channels]
    au_gt = np.array(au_gt)
    # print(len(au_gt))
    # length = int(au_gt.shape[0]*0.3)
    # au_gt = au_gt[-length:,:]
    # imu_dataset = imu_dataset[-length:,:,:]
    print(au_gt.shape)
    # assert emotion_au_gt.shape[0]==opt.au_len, "expression_len is wrong!"
    model_test(opt, imu_dataset, au_gt, device, criterion, model) # [expression_len, au_channels]
    # save
    # save_list = {"gt":gt_au,
    #              "pd":inference_au,
    #              "index":index_pred}
    # result_dir = os.path.join('./result/', opt.downstream_dataset[4:-4], opt.model_name)
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    # with open(result_dir + '/' + opt.downstream_dataset[4:-4] + '_' + opt.dataset_type + ".pkl", 'wb') as f:
    #     pickle.dump(save_list, f, protocol=4)
        # save数据
        # dataset_unity['gt'].append(emotion_au_gt)
        # dataset_unity['pd'].append(emotion_inference_au)

def main(opt):

    path_root = "../new_data_P14_auex/"
    device = torch.device(opt.cuda if torch.cuda.is_available() and opt.use_cuda else "cpu")
    print("gpu: ",device)
    print(opt.model_name)
    # time + channel
    for imu_file in os.listdir(path_root):
        if imu_file != opt.downstream_dataset:
            continue
        print(imu_file)
        # data = np.load(os.path.join(path_root, file_name),allow_pickle=True).item()
        if 'au' in imu_file:
            au_file = imu_file.replace('imu_au', 'au_au')
        else:
            au_file = imu_file.replace('imu_ex', 'au_ex')
        imu_data = pd.read_csv(os.path.join(path_root, imu_file), header=None).values
        au_data = pd.read_csv(os.path.join(path_root, au_file), header=None).values
        print(imu_data.shape)
        print(au_data.shape)
        # au_data = au_data[:20000,:]
        # imu_data = imu_data[:20000,:]
        imuInference(opt, imu_data, au_data, device,window_length=opt.imu_len)
    print(opt.total_length)
    print("---------------------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_run',type=str,default='14220112')
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument('--cuda',type=str,default='cuda:0')
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_length', type=int, default=5400)
    parser.add_argument('--model_name', type=str, default='14220112_pred15_0.5_1.5_tdo0.1_cdo0.5_im100_au30_ci12_co14_lr0.01_pl20_s4_re0_dm256_nh8_el6_dl6_dff512_within_cut_32_auex.pkl')
    parser.add_argument('--pred_len', type=int, default=15, help='pred au len of decoder')
    parser.add_argument('--user', type=str, default="wzf")
    parser.add_argument('--downstream_dataset', type=str, default='imu_au_wzf2.csv', help='pred au len of decoder')
    parser.add_argument('--au_len', type=int, default=30, help='decoder input length')
    # parser.add_argument('--dataset_type',type=str,default='withinall')

    parser.add_argument('--imu_len', type=int, default=100, help='encoder input length') 
    
    parser.add_argument('--c_in', type=int, default=12, help='input channels')
    parser.add_argument('--c_out', type=int, default=14, help='output channels')
    parser.add_argument('--lambda1', type=float, default=0.5, help='lambda1 loss parameter')
    parser.add_argument('--lambda2', type=float, default=1.5, help='lambda2 loss parameter')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=6, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--tf_dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--conv_dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--pos_e',type=str,default='encoder')
    parser.add_argument('--au_cor',type=lambda x: (str(x).lower() == 'true'), help='', default=False)
    
    args = parser.parse_args()
    # 只是预测推理
    # test(opt=args)
    # 整体
    main(opt=args)

# expression
    