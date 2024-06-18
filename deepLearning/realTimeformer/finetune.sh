if [ ! -d "./ft_logs" ]; then
    mkdir ./ft_logs
fi

# if [ ! -d "./logs/Terminal" ]; then
#     mkdir ./logs/Terminal
# fi

dataset_path=./dataset/
data_path=../data_P8_ex/
base_weights_dir=./log/04121229/yty/04121229_pred45_0.5_1.5_tdo0.1_cdo0.5_im200_au60_ci12_co14_lr0.01_pl20_s4_re0_dm256_nh8_el6_dl6_dff512_cross_cut_32_auex.pkl/weights
patch_len=5
stride=4
revin=0 # false 0 true 1
imu_len=200 # imu desired_length
au_len=60 #au
c_in=12 # 没用
c_out=14
d_model=256
n_heads=8
e_layers=6
d_layers=6
d_ff=512
tf_dropout=0.1
conv_dropout=0.5
user=yty
num_worker=8
train_epochs=200
batch_size=32
learning_rate=0.001
dataset_type=auex
total_people=14
pred_len=45
turns_num=1
user_mode=within # 拿within 的数据微调
spilt_mode=cut
time_run=04191219
lambda1=0.5
lambda2=1.5
is_fine_tune=1
is_init=0
cuda="0"
devices="0,1"
pos_e=encoder
# is-generate
python -u fine_tune.py \
    --dataset_path $dataset_path \
    --data_path $data_path\
    --base_weights_dir $base_weights_dir\
    --patch_len $patch_len\
    --stride $stride\
    --revin $revin\
    --imu_len $imu_len\
    --au_len $au_len\
    --turns_num $turns_num\
    --c_in $c_in\
    --c_out $c_out\
    --d_model $d_model\
    --n_heads $n_heads\
    --e_layers $e_layers\
    --d_layers $d_layers\
    --d_ff $d_ff\
    --conv_dropout $conv_dropout\
    --tf_dropout $tf_dropout\
    --train_epochs $train_epochs\
    --batch_size $batch_size\
    --patience 20\
    --num_worker $num_worker\
    --learning_rate $learning_rate\
    --user_mode $user_mode\
    --user $user\
    --pred_len $pred_len\
    --spilt_mode $spilt_mode\
    --time_run $time_run\
    --total_people $total_people\
    --cuda $cuda\
    --lambda1 $lambda1\
    --lambda2 $lambda2\
    --pos_e $pos_e\
    --is_init $is_init\
    --is_fine_tune $is_fine_tune\
    --au_cor False\
    --is_generate False\
    --use_multi_gpu False\
    --devices $devices\
    --dataset_type $dataset_type >ft_logs/$time_run'_'$user'_ft'$is_fine_tune'_tl'$turns_num'_in'$is_init'_'$lambda1'_'$lambda2'_im'$imu_len'_au'$au_len'_ci'$c_in'_co'$c_out'_lr'$learning_rate'_pl'$patch_len'_s'$stride'_re'$revin'_dm'$d_model'_nh'$n_heads'_el'$e_layers'_dl'$d_layers'_dff'$d_ff'_'$user_mode'_'$spilt_mode'_'$batch_size'_'$dataset_type.log 

