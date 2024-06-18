import torch
import pickle
import torch.nn as nn  
import argparse
from modelStruct import Transformer
import numpy as np
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


def channel_mae(y_pred, y_true):
    # 计算每个样本的每个通道的绝对误差
    absolute_errors = np.abs(y_true - y_pred)
    # 计算每个通道的平均绝对误差
    channel_maes = np.mean(absolute_errors, axis=(0, 1))
    mse_errors = (y_pred - y_true) ** 2
    channel_mses = np.mean(mse_errors, axis=(0, 1))
    return channel_maes, channel_mses
    
def test(args, model, loader, criterion, device, label, test_path = None):
    labels = []
    first_outputs = []
    second_outputs = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y,dec_in) in enumerate(loader):
            # batch y [batchsize, au len, channels]
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            dec_in = dec_in.float().to(device)

            dec_input = dec_in[:, 0, :]
            dec_input = dec_input.unsqueeze(0)
            for _ in range(batch_y.shape[1]-1):
                output_conv, output_former, dec_self_attns, dec_enc_attns = model(batch_x, dec_input)
                last_predicted_value = output_former[:,-1,:]
                last_predicted_value = last_predicted_value.unsqueeze(0)
                dec_input = torch.cat((dec_input, last_predicted_value.detach()), dim=1)
            # first: take model output 
            output_conv, final_prediction, dec_self_attns, dec_enc_attns = model(batch_x, dec_input)
            last_predicted_value = final_prediction[:,-1,:]
            last_predicted_value = last_predicted_value.unsqueeze(0)
            # seconde: take every point prediction
            dec_input = torch.cat((dec_input, last_predicted_value.detach()), dim=1)
            second_final_pred = dec_input[:,1:,:]

            
            final_prediction = final_prediction.detach().cpu()
            second_final_pred = second_final_pred.detach().cpu()
            batch_y = batch_y.detach().cpu()
            

            first_outputs.append(final_prediction)
            second_outputs.append(second_final_pred)
            labels.append(batch_y)

            if i % 2000 == 0:
                # input = batch_x.detach().cpu().numpy()
                gt = batch_y[0, :, :]
                pd = second_final_pred[0, :,:]
                visual_pred_lens(second_final_pred.shape[1], args.c_out, true=gt, preds=pd, name=os.path.join(test_path, str(i) + '.pdf'))

    first_outputs = torch.stack(first_outputs, dim=0).numpy()
    second_outputs = torch.stack(second_outputs, dim=0).numpy()
    labels = torch.stack(labels, dim=0).numpy()
    print(first_outputs.shape)

    first_outputs = first_outputs.reshape(-1, first_outputs.shape[-2], first_outputs.shape[-1])
    second_outputs = second_outputs.reshape(-1, second_outputs.shape[-2], second_outputs.shape[-1])
    labels = labels.reshape(-1, labels.shape[-2], labels.shape[-1])
    print(first_outputs.shape)
    # -------------
    # first:
    mae, mse, rmse, rse, corr = metric(first_outputs, labels)
    channel_maes, channel_mses = channel_mae(first_outputs, labels)
    print('Test mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
    print('Test channels mae:{}'.format(channel_maes))
    print('Test channels mse:{}'.format(channel_mses))
    # -----------
    # second:
    mae, mse, rmse, rse, corr = metric(second_outputs, labels)
    channel_maes, channel_mses = channel_mae(second_outputs, labels)
    print('Test mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
    print('Test channels mae:{}'.format(channel_maes))
    print('Test channels mse:{}'.format(channel_mses))
    
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
    
    model = Transformer(conv_dropout=args.conv_dropout,
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
                        is_inference=True,
                        au_cor_matrix=None,
                        is_au_cor=args.au_cor)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    model.eval()
    return criterion, model
def main(opt):
    print(f'test_path: {opt.test_dataset_path}')
    print(f'model_name: {opt.model_name}')
    test_path = os.path.join(opt.result_saving_path, opt.time_run, opt.user, opt.model_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    # load data
    device = torch.device(opt.cuda if torch.cuda.is_available() and opt.use_cuda else "cpu")
    criterion, model = initModel(opt,device)
    print(device)

    # train_dataset = My_Dataset(path=opt.test_dataset_path, label='train')
    # val_dataset = My_Dataset(path=opt.test_dataset_path, label='val')
    test_dataset = My_Dataset(path=opt.test_dataset_path, label='test')

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=False,
    #                                 num_workers=opt.num_worker, drop_last=False)
    # vali_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
    #                                         num_workers=opt.num_worker, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False,
                                            num_workers=opt.num_worker, drop_last=False)
    print(f"len of test loader{len(test_loader)}")
    
    # test(opt, model, train_loader, criterion, device, 'train',test_path)
    # print("---------------------------------------------------------")
    # test(opt, model, vali_loader, criterion, device, 'val',test_path)
    # print("---------------------------------------------------------")
    test(opt, model, test_loader, criterion, device, 'test',test_path)
    print("---------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_run',type=str,default='04261224')
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument('--cuda',type=str,default='cuda:0')
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_dataset_path', type=str, default="")
    parser.add_argument('--model_name', type=str, default='04261224_pred45_0.5_1.5_tdo0.1_cdo0.5_im200_au60_ci12_co14_lr0.01_pl20_s4_re0_dm256_nh8_el6_dl6_dff512_within_cut_32_auex.pkl')
    parser.add_argument('--pred_len', type=int, default=45, help='pred au len of decoder')
    parser.add_argument('--test_path', type=str, default='ex_P20_within_cut_w200_fxc.pkl')
    parser.add_argument('--user', type=str, default="cyx")
    parser.add_argument('--result_saving_path', type=str, default="")
    

    parser.add_argument('--imu_len', type=int, default=200, help='encoder input length') 
    parser.add_argument('--au_len', type=int, default=60, help='decoder input length')
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
    