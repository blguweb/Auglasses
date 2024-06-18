if [ ! -d "./test_logs" ]; then
    mkdir ./test_logs
fi

# if [ ! -d "./logs/Terminal" ]; then
#     mkdir ./logs/Terminal
# fi
time_run=04121653  # model 的编码
user=oyh
type_=crossuser
result_saving_path=./crossuser_result/
# ../datasetauexs1/auexs1_P14_within_cut_w60_s1_oyh_200/
test_dataset_path=../dataset_P14_auexs1/auexs1_P14_within_cut_w60_s15_oyh_200
model_name=04121653_pred45_0.5_2.0_tdo0.1_cdo0.5_im200_au60_ci12_co14_lr0.01_pl20_s4_re0_dm256_nh8_el6_dl6_dff512_cross_cut_32_auex.pkl
test_path=auexs1_P14_within_cut_w60_s15_oyh_200
imu_len=200 # imu desired_length
au_len=60 #au
pred_len=45
c_in=12 # 没用
c_out=14
d_model=256
lambda1=0.5
lambda2=1.5
n_heads=8
e_layers=6
d_layers=6
d_ff=512
tf_dropout=0.1
conv_dropout=0.5

num_worker=1
batch_size=1

cuda=cuda:0
pos_e=encoder
# is-generate
python -u test.py \
    --test_dataset_path $test_dataset_path \
    --model_name $model_name\
    --imu_len $imu_len\
    --user $user\
    --result_saving_path $result_saving_path\
    --au_len $au_len\
    --c_in $c_in\
    --test_path $test_path\
    --c_out $c_out\
    --d_model $d_model\
    --n_heads $n_heads\
    --e_layers $e_layers\
    --d_layers $d_layers\
    --d_ff $d_ff\
    --conv_dropout $conv_dropout\
    --tf_dropout $tf_dropout\
    --batch_size $batch_size\
    --num_worker $num_worker\
    --time_run $time_run\
    --cuda $cuda\
    --lambda1 $lambda1\
    --lambda2 $lambda2\
    --pos_e $pos_e\
    --au_cor False\
    --pred_len $pred_len >test_logs/$type_'_'$test_path'__'$model_name.log 

