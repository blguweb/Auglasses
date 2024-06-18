if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# if [ ! -d "./logs/Terminal" ]; then
#     mkdir ./logs/Terminal
# fi

dataset_path=./withindataset/
data_path=../data_P8_ex/
patience=10
stride=4
patch_len=20
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
user=ssc
num_worker=4
train_epochs=200
batch_size=32
learning_rate=0.01
dataset_type=auex
total_people=14
pred_len=45
user_mode=within
spilt_mode=cut
time_run=04261224
lambda1=0.5
lambda2=1.5
cuda="0"
devices="0,1"
pos_e=encoder
# is-generate
# for user in wzf yty yzh lls
# do
python -u exp_main.py \
    --dataset_path $dataset_path \
    --data_path $data_path\
    --patch_len $patch_len\
    --stride $stride\
    --revin $revin\
    --imu_len $imu_len\
    --au_len $au_len\
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
    --patience $patience\
    --num_worker $num_worker\
    --learning_rate $learning_rate\
    --user_mode $user_mode\
    --spilt_mode $spilt_mode\
    --time_run $time_run\
    --total_people $total_people\
    --cuda $cuda\
    --lambda1 $lambda1\
    --lambda2 $lambda2\
    --pos_e $pos_e\
    --user $user\
    --au_cor False\
    --is_generate False\
    --use_multi_gpu False\
    --pred_len $pred_len\
    --devices $devices\
    --dataset_type $dataset_type >logs/$time_run'_'$user'_pred'$pred_len'_cdo'$conv_dropout'_fdo'$tf_dropout'_'$lambda1'_'$lambda2'_im'$imu_len'_au'$au_len'_ci'$c_in'_co'$c_out'_lr'$learning_rate'_pl'$patch_len'_s'$stride'_re'$revin'_dm'$d_model'_nh'$n_heads'_el'$e_layers'_dl'$d_layers'_dff'$d_ff'_'$user_mode'_'$spilt_mode'_'$batch_size'_'$dataset_type.log 
# done
