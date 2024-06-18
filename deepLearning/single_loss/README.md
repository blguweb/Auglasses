# convformerTimeSerial
之前是采用convformer架构 由imu预测au
现在添加前期已经预测过的imu的结果；也就是输入包括已经预测的imu时间段和未经过预测的imu时间段；
然后预测整体au，而且ender_result采用 已经预测au_len辅助，后期的au_len 进行mask

数据的结构也是 [au_len ,channel] , [au_len,imu_len, imu_channel]

# 04061541 test within session

# 04061720 enc dec反过来 

# 04061427 在04061720集成上把encoder的 n_layer = 1

# 04061720 pred_len=0 是对照实验，enc没有gt输入，但是预测还是后面15个;可以达到0.11mae

fine tune
04181218 train / 7 * turns ; val: 93 
04181219  turns / 7 *1 val/2
04181220 turns  / 7*1  val/4
04181221 train /10 * turns   val/4
越多数据越好；
04181222 train /20 * turns   val/4 ok
暂时用这个；
04181223 train /32 * turns   val/5
04181224 train /10 * turns   val/4


04190905 cut_fintue 1:自己训练自己验证；
04190915 cut_fintue 2：自己训练自己验证； 
04191023 尝试用所有train进行训练和验证；
04191215 自己训练自己验证；dagou；
04191218 自己训练自己验证；数据量/5 
04191219 自己训练自己验证；数据量/2




# 最好的微调：
很牛！用lls的cross模型去微调其他人的模型；效果很好! 不行；因为数据泄露了
04181745

# 04191225
04191225 数据量太少容易预测成直线；rl 0.0001
04191705 只用happy
04191707


# train.sh 04191602 新的数据集进行训练！
14191852 数据集步进为6

# grid
14220112 history 15 : pred_len 15 30 45 60 


# freq
14220050 50hz
14220100 100hz
14220201 200hz
14220300 300hz


