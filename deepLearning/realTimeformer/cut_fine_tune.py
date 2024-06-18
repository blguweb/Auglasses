import torch.nn as nn
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import random
import argparse
import numpy as np
import time
from utils import *
from modelStruct import Convformer
import matplotlib.pyplot as plt
from metrics import metric
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

class IMU_Dataset(Dataset):
    """自定义数据集"""

    def __init__(self, data_x, data_y):
        self.x = data_x
        self.y = data_y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        label = np.delete(self.y[item],(9,11,16), axis=1)
        return self.x[item], label
    
# def test(args, file_name, file_path, nw, device):
#     path_label = '{}_ft{}_in{}_{}_{}_tdo{}_cdo{}_im{}_au{}_ci{}_co{}_lr{}_pl{}_s{}_re{}_dm{}_nh{}_el{}_dl{}_dff{}_{}_{}_{}_{}.pkl'.format(
#         args.time_run,
#         args.is_fine_tune,
#         args.is_init,
#         args.lambda1,
#         args.lambda2,
#         args.tf_dropout,
#         args.conv_dropout,
#         args.imu_len,
#         args.au_len,
#         args.c_in,
#         args.c_out,
#         args.learning_rate,
#         args.patch_len,
#         args.stride,
#         args.revin,
#         args.d_model,
#         args.n_heads,
#         args.e_layers,
#         args.d_layers,
#         args.d_ff,
#         args.user_mode,
#         args.spilt_mode,
#         args.batch_size,
#         args.dataset_type
#     )
#     test_path = os.path.join('./finetune_result/', args.time_run, args.user, path_label)
#     if not os.path.exists(test_path):
#         os.makedirs(test_path)
#     # 导入 au_cor_matrix矩阵
#     au_cor_matrix = np.load("au_cor_matrix.npy", allow_pickle=True)
#     au_cor_matrix = torch.from_numpy(au_cor_matrix).float().to(device)
#     model = Convformer(conv_dropout=args.conv_dropout,
#                         dec_dropout=args.tf_dropout,
#                         enc_dropout=args.tf_dropout,
#                         n_heads=args.n_heads,
#                         e_layers=args.e_layers,
#                         d_layers=args.d_layers,
#                         d_ff=args.d_ff,
#                         d_model=args.d_model,
#                         au_len=args.au_len,
#                         imu_len=args.imu_len,
#                         c_out=args.c_out,
#                         pos_e=args.pos_e,
#                         au_cor_matrix=None,
#                         is_au_cor=args.au_cor)
#     best_model_path = args.base_weights_dir + '/' + 'checkpoint.pth'
#     model.load_state_dict(torch.load(best_model_path))

#     if args.use_multi_gpu and args.use_cuda:
#         model = nn.DataParallel(model, device_ids=args.device_ids)
#     model = model.to(device)

#     train_dataset = My_Dataset(imu_data,au_data)
#     val_dataset = My_Dataset(path=file_path, label='val')
#     test_dataset = My_Dataset(path=file_path, label='test')
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
#                                     num_workers=nw, drop_last=True)
#     vali_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
#                                             num_workers=nw, drop_last=True)
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
#                                             num_workers=nw, drop_last=True)

#     print(f"len of train loader: {len(train_loader)}, test loader{len(test_loader)}, val loader{len(vali_loader)}")
    
#     criterion = nn.MSELoss()

#     total_loss = []
#     result_loss = []
#     labels = []
#     outputs = []
#     model.eval()
#     with torch.no_grad():
#         for i, (batch_x, batch_y) in enumerate(test_loader):
#             batch_x = batch_x.float().to(device)
#             batch_y = batch_y.float().to(device)
#             dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
#             dec_inp = torch.cat([batch_y[:, :(args.au_len - args.pred_len), :], dec_inp], dim=1).float().to(device)

#             output_conv, output_former, dec_self_attns, dec_enc_attns = model(batch_x, dec_inp)

#             # outputs = outputs[:, -self.args.pred_len:, f_dim:]
#             # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
#             label1 = torch.reshape(batch_y,(batch_y.shape[0] * batch_y.shape[1], batch_y.shape[2]))

                
            
#             output_conv = output_conv.detach().cpu()
#             output_former = output_former.detach().cpu()
#             batch_y = batch_y.detach().cpu()
#             label1 = label1.detach().cpu()
            
#             pred = output_former[:, -args.pred_len:, :]
#             true = batch_y[:, -args.pred_len:, :]

#             loss1 = criterion(output_conv, label1)
#             loss2 = criterion(pred, true)
#             loss = args.lambda1 * loss1 + args.lambda2 * loss2
#             # loss = criterion(pred, true)

#             result_loss.append(loss2)
#             total_loss.append(loss)
#             outputs.append(output_former)
#             labels.append(batch_y)
#             if i % 10 == 0:
#                 # input = batch_x.detach().cpu().numpy()
#                 gt = batch_y[3, :]
#                 pd = output_former[3, :]
#                 visual_pred_lens(args.au_len, args.c_out, true=gt, preds=pd, name=os.path.join(test_path, str(i) + '.pdf'))
#     result_loss = np.average(result_loss)
#     outputs = torch.stack(outputs, dim=0).numpy()
#     labels = torch.stack(labels, dim=0).numpy()
#     outputs = outputs.reshape(-1, outputs.shape[-2], outputs.shape[-1])
#     labels = labels.reshape(-1, labels.shape[-2], labels.shape[-1])
#     mae, mse, rmse, rse, corr = metric(outputs, labels)
#     channel_maes, channel_mses = channel_mae(outputs, labels)
#     print('Test: result loss:{}, mse:{}, mae:{}, rse:{}'.format(result_loss,mse, mae, rse))
#     print('Test channels mae:{}'.format(channel_maes))
#     print('Test channels mse:{}'.format(channel_mses))
#     total_loss = np.average(total_loss)
#     print(f"Test total loss:{total_loss}")
#     # model.train()
#     # return total_loss

def vali(args, model, vali_loader, criterion, device, label, test_path = None):
    total_loss = []
    result_loss = []
    labels = []
    outputs = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :(args.au_len - args.pred_len), :], dec_inp], dim=1).float().to(device)

            # encoder - decoder

            output_conv, output_former, dec_self_attns, dec_enc_attns = model(batch_x, dec_inp)

            # outputs = outputs[:, -self.args.pred_len:, f_dim:]
            # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
            label1 = torch.reshape(batch_y,(batch_y.shape[0] * batch_y.shape[1], batch_y.shape[2]))

                
            
            output_conv = output_conv.detach().cpu()
            output_former = output_former.detach().cpu()
            batch_y = batch_y.detach().cpu()
            label1 = label1.detach().cpu()
            
            pred = output_former[:, -args.pred_len:, :]
            true = batch_y[:, -args.pred_len:, :]

            loss1 = criterion(output_conv, label1)
            loss2 = criterion(pred, true)
            loss = args.lambda1 * loss1 + args.lambda2 * loss2
            # loss = criterion(pred, true)

            result_loss.append(loss2)
            total_loss.append(loss)
            outputs.append(output_former)
            labels.append(batch_y)
            if label == "test":
                
                if i % 2 == 0:
                    # input = batch_x.detach().cpu().numpy()
                    gt = batch_y[0, :, :]
                    pd = output_former[0, :, :]
                    visual_pred_lens(args.au_len, args.c_out, true=gt, preds=pd, name=os.path.join(test_path, str(i) + '.pdf'))
    result_loss = np.average(result_loss)
    outputs = torch.stack(outputs, dim=0).numpy()
    labels = torch.stack(labels, dim=0).numpy()
    outputs = outputs.reshape(-1, outputs.shape[-2], outputs.shape[-1])
    labels = labels.reshape(-1, labels.shape[-2], labels.shape[-1])
    mae, mse, rmse, rse, corr = metric(outputs, labels)
    channel_maes, channel_mses = channel_mae(outputs, labels)
    if label == "test":
        print('Test: result loss{}, mse:{}, mae:{}, rse:{}'.format(result_loss,mse, mae, rse))
    elif label == "vali":
        print('Vali: result loss{}, mse:{}, mae:{}, rse:{}'.format(result_loss,mse, mae, rse))
    print('channels mae:{}'.format(channel_maes))
    print('channels mse:{}'.format(channel_mses))
    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def fine_tune(args, train_data, train_label, val_data, val_label, nw, device):
    path_label = '{}_ft{}_in{}_{}_{}_tdo{}_cdo{}_im{}_au{}_ci{}_co{}_lr{}_pl{}_s{}_re{}_dm{}_nh{}_el{}_dl{}_dff{}_{}_{}_{}_{}.pkl'.format(
        args.time_run,
        args.is_fine_tune,
        args.is_init,
        args.lambda1,
        args.lambda2,
        args.tf_dropout,
        args.conv_dropout,
        args.imu_len,
        args.au_len,
        args.c_in,
        args.c_out,
        args.learning_rate,
        args.patch_len,
        args.stride,
        args.revin,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.user_mode,
        args.spilt_mode,
        args.batch_size,
        args.dataset_type
    )
    log_path = os.path.join('./finetune_log/',args.time_run, args.user, str(args.turns_num), path_label)
    # 创建日志文件
    tb_writer = SummaryWriter(log_dir=log_path)
    # 导入 au_cor_matrix矩阵
    # au_cor_matrix = np.load("au_cor_matrix.npy", allow_pickle=True)
    # au_cor_matrix = torch.from_numpy(au_cor_matrix).float().to(device)
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
    best_model_path = args.base_weights_dir + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    # 冻结除最后的全连接层和解码器以外的所有层
    # for name, param in model.named_parameters():
    #     print("name",name)
    #     if 'projection' not in name:  # 指定不冻结的层名关键字
    #         param.requires_grad = False

    # # 打印模型的参数状态，确认哪些参数被冻结
    # for name, param in model.named_parameters():
    #     print(f"{name} is {'not ' if param.requires_grad else ''}frozen")
    
    # print(model.projection)
    # 如果需要，重新初始化最后一层的权重
    if args.is_init:
        if hasattr(model, 'projection'):  # 假设最后一层是fc2
            # 使用适当的初始化方法，这里使用kaiming初始化作为示例
            nn.init.kaiming_normal_(model.projection.weight)
            if model.fc2.bias is not None:
                nn.init.constant_(model.projection.bias, 0)


    if args.use_multi_gpu and args.use_cuda:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    model = model.to(device)

    train_dataset = IMU_Dataset(train_data, train_label)
    val_dataset = IMU_Dataset(val_data, val_label)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=nw, drop_last=True)
    vali_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=nw, drop_last=True)

    print(f"len of train loader: {len(train_loader)}, val loader{len(vali_loader)}")
    


    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    model_optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=0.9) 
    # torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9) 
    # optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()


    scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                        steps_per_epoch = train_steps,
                                        pct_start = args.pct_start,
                                        epochs = args.train_epochs,
                                        max_lr = args.learning_rate)
    # dir 
    weights_dir = os.path.join('./finetune_log/', args.time_run, args.user, str(args.turns_num), path_label,"weights")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    test_path = os.path.join('./finetune_result/', args.time_run, args.user, str(args.turns_num),path_label)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        count_ = 0
        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader): # [batch, au_len,labels/imu_len,channel]
            iter_count += 1
            count_ += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            # encoder - decoder
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :(args.au_len - args.pred_len), :], dec_inp], dim=1).float().to(device)


            output_conv, output_former, dec_self_attns, dec_enc_attns = model(batch_x, dec_inp)

            # outputs = outputs[:, -self.args.pred_len:, f_dim:]
            # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
            pred = output_former[:, -args.pred_len:, :]
            true = batch_y[:, -args.pred_len:, :]
                
            label1 = torch.reshape(batch_y,(batch_y.shape[0] * batch_y.shape[1], batch_y.shape[2]))
            loss1 = criterion(output_conv, label1)
            loss2 = criterion(pred, true)
            loss = args.lambda1 * loss1 + args.lambda2 * loss2
            train_loss.append(loss.item())
            # print(f"loss1: {loss1.item()}, loss2: {loss2.item()}")

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()


            loss.backward()
            model_optim.step()
                
        print("COUNT", count_)
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = vali(args, model, train_loader, criterion, device, "test",test_path=test_path)
        # test_loss = vali(args, model, test_loader, criterion, device, "test",test_path=test_path)# test visual
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))
        tags = ["train_loss",  "val_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)

        tb_writer.add_scalar(tags[1], vali_loss, epoch)

        tb_writer.add_scalar(tags[2], model_optim.param_groups[0]["lr"], epoch)
        
        early_stopping(vali_loss, model, weights_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # 没有 schedule step
        adjust_learning_rate(model_optim, scheduler, epoch + 1, args)

    # best_model_path = weights_dir + '/' + 'checkpoint.pth'
    # model.load_state_dict(torch.load(best_model_path))


def cutting_segment(opt, imu, au,):
    val_x = None
    val_y = None
    train_x = None
    train_y = None
    for em_sam in range(au.shape[0]):
        imu_lists = []
        au_lists = []
        # making dataset
        for au_index in range(au.shape[1]):
            imu_m_index = find_index(au[em_sam, au_index, 0], imu[em_sam, :, 0])
            if imu_m_index == None:
                continue
            imu_start_index = imu_m_index - opt.imu_len//2
            imu_end_index = imu_start_index + opt.imu_len
            if imu_end_index >= imu.shape[1] or imu_start_index < 0:
                continue
            # Selecting the imu data in the window
            imu_lists.append(imu[em_sam, imu_start_index:imu_end_index, 1:])
            au_lists.append(au[em_sam, au_index, 1:])
        # 
        imu_lists = np.array(imu_lists) # [expression_len, window_length, imu_channels]
        au_lists = np.array(au_lists)

        X = []
        Y = []
        # A = []
        # 从AU数据开始滑动窗口
        for slid_index in range(0, len(au_lists), 15):

            end_index = slid_index + opt.au_len
            if end_index > len(au_lists):
                break

            # 获取当前窗口的AU数据
            current_au_window = au_lists[slid_index:end_index]
            
            current_imu_units = imu_lists[slid_index:end_index]


            X.append(np.array(current_imu_units))
            Y.append(np.array(current_au_window))
        X = np.array(X)
        Y = np.array(Y)
        if em_sam == au.shape[0]-1:
            # val_dataset
            if val_x is None:
                val_x = X
                val_y = Y
            else:
                val_x = np.vstack((val_x, X))
                val_y = np.vstack((val_y, Y))
        else:
            # train_dataset
            if train_x is None:
                train_x = X
                train_y = Y
            else:
                train_x = np.vstack((train_x, X))
                train_y = np.vstack((train_y, Y))
    # train_y = np.array(train_y)
    # train_x = np.array(train_x)
    # val_x = np.array(val_x)
    # val_y = np.array(val_y)
    # train_y = train_y.reshape(-1,train_y.shape[2],train_y.shape[3])
    # train_x = train_x.reshape(-1,train_x.shape[2],train_x.shape[3], train_x.shape[4])
    # val_x = val_x.reshape(-1,val_x.shape[2],val_x.shape[3], val_x.shape[4])
    # val_y = val_y.reshape(-1,val_y.shape[2],val_y.shape[3])
    return train_x, train_y, val_x, val_y

    # return model
def general_finetune_dataset(opt):
    emotions = ["none","happy","sad","fear","angry","surprise","disgusted"]
    dataset_dict = {emotion: [] for emotion in emotions}
    path_root = "./ft_dataset/"
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    for file_name in os.listdir(path_root):
        # if opt.user not in file_name:
        #     continue
        data = np.load(os.path.join(path_root, file_name),allow_pickle=True).item()
        print(file_name)

        for emotion, value in data.items():
            print(emotion, len(value['au']))
            if emotion not in ["happy"]:
                continue
            if value['imu'] != [] and value['au'] != []:
                # print(data)
                imu_data = np.array(value['imu'], dtype = float) # [n_samples, window_len, channels]
                au_data = np.array(value['au'], dtype = float) # [n_samples, window_len, channels]
                
                # 拿n轮的数据进行生成数据集；
                imu_data = imu_data[:opt.turns_num,:]
                au_data = au_data[:opt.turns_num,:]
                t_x,t_y,v_x,v_y = cutting_segment(opt, imu_data, au_data)

                
                if train_x is None:
                    train_x = t_x
                    train_y = t_y
                    val_x = v_x
                    val_y = v_y
                else:
                    train_x = np.vstack((train_x, t_x))
                    train_y = np.vstack((train_y, t_y))
                    val_x = np.vstack((val_x, v_x))
                    val_y = np.vstack((val_y, v_y))
            else:
                print(file_name, emotion, "data is empty!")
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    print(f'train len: {train_x.shape}, {train_y.shape}, val len: {val_x.shape}, {val_y.shape}')
    return train_x, train_y, val_x, val_y



def main(opt):
    # 1.读取一些配置参数，并且输出
    print('Args in experiment:')
    print(opt)

    # 设备
    print("gpu_count",torch.cuda.device_count())
    opt.use_cuda = True if torch.cuda.is_available() and opt.use_cuda else False
    
    if opt.use_cuda and opt.use_multi_gpu:
        opt.dvices = opt.devices.replace(' ', '')
        device_ids = opt.devices.split(',')
        print(device_ids)
        opt.device_ids = [int(id_) for id_ in device_ids]
        opt.cuda = device_ids[0] # 主GPU
    if opt.use_cuda:
        device = torch.device('cuda:{}'.format(opt.cuda))
    else:
        device = torch.device('cpu')
        print('Use CPU')

    
    print("cpu_count",os.cpu_count())
    nw = min([os.cpu_count(), opt.batch_size, opt.num_worker if opt.batch_size > 1 else 0])
    print('Using {} dataloader workers every process'.format(nw))

    # if opt.is_generate:
    #     general_dataset(user_mode=opt.user_mode, dataset_type=opt.dataset_type,
    #                     total_people=opt.total_people, time_run=opt.time_run,
    #                     spilt_mode=opt.spilt_mode, test_rate=0.2, val_rate=0.1,
    #                     window_length=opt.window_length,
    #                     data_root=opt.data_path,dataset_root=opt.dataset_path)
        

    train_x, train_y, val_x, val_y = general_finetune_dataset(opt)

    start_name = '{}_P{}_{}_{}_w{}'.format(
        opt.dataset_type,
        opt.total_people,
        opt.user_mode,
        opt.spilt_mode,
        opt.au_len
    )
    if opt.is_fine_tune:
        fine_tune(opt, train_x, train_y, val_x, val_y, nw, device)
    # if opt.user_mode == 'within':
    #     for file_name in os.listdir(opt.dataset_path):
    #         if file_name.startswith(start_name) and file_name.endswith(str(opt.imu_len)):
    #             if opt.user in file_name:
    #             # 构造完整的文件路径
    #                 file_path = os.path.join(opt.dataset_path, file_name)
    #                 print(file_name)

    #                 else:
    #                     # 在已有模型基础上测试另外一个人的数据
    #                     test(opt, file_name, file_path, nw, device)
    #             # loadData(opt, file_name, file_path,nw, device)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer for Time Series Forecasting')
    # random seed
    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')
    # path
    parser.add_argument('--dataset_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str,  default='./data/', help='data file')
    parser.add_argument('--base_weights_dir', type=str,  default='../conv_trans_cor/log/04021631/04021631_0.3_1.5_im200_au60_ci12_co13_lr0.0001_pl20_s4_re0_dm256_nh8_el6_dl6_dff512_within_cut_32_auex.pkl/weights/', help='data file')
    # parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # forecasting task
    # parser.add_argument('--seq_len', type=int, default=96, help='imu input sequence length')
    # parser.add_argument('--label_len', type=int, default=48, help='start token length')
    # parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # Transformer
    parser.add_argument('--patch_len', type=int, default=20, help='patch length')
    parser.add_argument('--stride', type=int, default=4, help='stride')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--imu_len', type=int, default=800, help='encoder input length') 
    parser.add_argument('--au_len', type=int, default=60, help='decoder input length')
    parser.add_argument('--c_in', type=int, default=12, help='input channels')
    parser.add_argument('--c_out', type=int, default=17, help='output channels')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--tf_dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--conv_dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--output_attention', action='store_false', help='whether to output attention in ecoder')
    parser.add_argument('--pos_e',type=str,default='encoder')
    parser.add_argument('--au_cor',type=lambda x: (str(x).lower() == 'true'), help='', default=False)
    parser.add_argument('--lambda1', type=float, default=0.5, help='lambda1 loss parameter')
    parser.add_argument('--lambda2', type=float, default=0.5, help='lambda2 loss parameter')
    parser.add_argument('--is_init', type=int, default=1, help='init or base weight during fine tune; True 1 False 0')
    parser.add_argument('--is_fine_tune', type=int, default=1, help='True 1 False 0')
    parser.add_argument('--pred_len', type=int, default=1, help='pred au len of decoder')
    parser.add_argument('--user',type=str,default='lyr')
    parser.add_argument('--turns_num', type=int, default=7, help='')
    
    # optimization
    parser.add_argument('--num_worker', type=int, default=16, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--dataset_type',type=str,default='ex_seg3')
    parser.add_argument('--total_people', type=int, default=8, help='total people of dataset')
    parser.add_argument('--user_mode', type=str, default='cross')
    parser.add_argument('--spilt_mode',type=str,default='cut')
    parser.add_argument('--time_run',type=str,default='03071719')
    parser.add_argument('--optimizer', type=str, default='adamw')  # sgd,adam,adamw
    parser.add_argument('--is_generate',type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start') # 学习率上升部分所占比例
    
    # GPU
    parser.add_argument('--use_cuda', type=bool, default=True, help='use gpu')
    parser.add_argument('--cuda',type=str,default='0')
    parser.add_argument('--use_multi_gpu',  type=lambda x: (str(x).lower() == 'true'), help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='4,6', help='device ids of multile gpus')
    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    main(opt=args)