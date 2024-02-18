

baseline 均值！
!!! 
得用3秒滤波 不可以用1秒滤波！
# selectSize
## 确定c和b
- 数据:
  - imu_c_15_50/ index_c_15_50/ time_c_15_50 命名格式：1.5=d 50 * 2 = e;c表示探究的维度；
  - index_xx：表示表情顺序的索引表；time_xx：表示表情epochs的时间戳；
- 代码:SNR_data_collect.py


c_15_55_0放置胶没有完全张开；放置在垃圾桶

@@ 左边信号的强度
c_20_50可以看左边的信号情况
c_15_50；可以看左边的信号情况

## 流程
1、去掉异常值
2、低通滤波
3、切片
4、基线去除，不可以用低通，把信息去掉了 均值；
5、计算RMS/SNR
(没有去除伪影)

# 伪影的探究
1、控制眼镜不运动；
2、自由运动的数据；
方法：
查看波形；
相关分析；

数据：c_15_50_static
代码：artifactsAnalyse.py

明天加上表情的标签 以及 这是自由运动的；

&& 采用第一段静止进行映射，曲线趋势是一样，但是会出现基线的漂移；
观察不同表情的相关性；特别是大动作；简称

相关度的分析；：有必要做?
头部伪影 需要的；可能是信号长度太长了，关键只有一小部分使得数据特征被淹没，看不出
==需要确定不同人的伪影是否大概一致；==

==再度思考z的方向是否需要反方向；映射后到底代表了什么==
简称

# SNR
先不进行归一化；acc <0 ryro;把acc 弄成》1？
1：1：归一化
先不和；分开看，以3为界限
重新采集15—50的数据

## 新旧
c_15_45是旧的软胶；
c_15_45是新的软胶；


##d_10_3很紧

d_8是没有son磨具 d_8 d_9
d_12_2没有垂下眼睛
d——8———4sad 没有垂下眼睛；只是采集了第一组难过表情，其他表情无效；

h1 vhorzion 骨头上面


# 江之行数据
big2有效 而且时长缩短为1s
noModify 第一个静止眨眼了
完整数据在外面，细节数据在里面；

# lyr_left
左边的软胶更换成了整体式结构，测试左边软胶的性能

# artifacts - 副本.py
测试新的expression的形状


# code

user_record.py: 进行用户数据录入，进行calibration 弃用


采用snr_data_collect.py进行数据采集，然后进行相关性分析；
然后correlate_caculate.py: 计算相关性的函数
imu_collect.py: 采集imu数据

Spectru.py是imu信号的时频图
calibra_signal.npy是calibration的时候配对的calibration信号
目前使用1112目录下的calira.py
calibration文件下的mmd也是类似 的代码，但是没有使用
MMD_副本.py生成calibra_signal.npy

# using code
## 前期准备
  MMD_副本.py生成calibra_signal.npy 
## 采集数据集
  imu_collect.py
  corrate_caculate.py
  SNR_data_collect.py
## 处理数据
  deeplearning/GenerateData.py 将au和imu数据整体时间段对齐；然后作为数据集；
  查看数据集的良好情况
  - corrate_caculate.py
  - calibration_data_presee.py
  - video_align_imu_signal
  

# dataset
@ 
pre_sys1 - ex_sys1 - au_sys1 - video
pre_sys1 - ex_sys2 - au_sys2 - ok

@ 他的信号没有通过calibration
pre_yl3 - ex_yl1 - au_yl1
pre_yl3 - ex_yl2 - au_yl2

@第一次通过calibration 第二次不太能通过calibration
pre_yh3 - ex_yh1 - au_yh1
pre_yh7/4 - ex_yh2 - au_yh2

@ 废弃 wgw
@第一次通过calibration 第二次不太能通过calibration
pre_czf2 - ex_czf1 - au_czf1
pre_czf5 - ex_czf2 - au_czf2

@ 废弃 yjy1


@lyr
pre_lyr - ex_lyr - au_lyr

@ nxq
nxq3 - ex_nxq1 - au_nxq1
pre_nxq4 - ex_nxq2 - au_nxq2

@ dll
pre_dll1- ex_dll1 - au_dll1
pre_dll2- ex_dll2 - au_dll2

@lls
pre_lls1- ex_lls1 - au_lls1
pre_lls2 - ex_lls2 - au_lls2

@ mrq
pre_mrq3 - ex_mrq1 - au_mrq1
pre_mrq4 - ex_mrq2 - au_mrq2

@ fb
pre_fb1 - ex_fb1 - au_fb1
pre_fb2 - ex_fb2 - au_fb2

@ dyw
pre_dyw1 - ex_dyw1 - au_dyw1
pre_dyw4 - ex_dyw2 - au_dyw2

@ yzh
pre_yzh1 - ex_yzh1 - au_yzh1
pre_yzh2 - ex_yzh2 - au_yzh2

@ wzf
pre_wzf1 - ex_wzf1 - au_wzf1
pre_wzf2 - ex_wzf2 - au_wzf2

@ ssr
pre_ssr1 - ex_ssr1 - au_ssr1
pre_ssr2 - ex_ssr2 - au_ssr2

@ fxc
pre_fxc2 - ex_fxc1 - au_fxc1
pre_fxc3 - ex_fxc2 - au_fxc2
ex_fxc2掉帧；

@cyx
pre_cyx3 - ex_cyx1 - au_cyx1
pre_cyx4 - ex_cyx2 - au_cyx2




@ ssc
最后一行时间更改，当时突然断了，因为只会提取最开始和最后的时间，所以把最后一行的时间改成了视频允许的时间
原：15:14:12.511,15:14:17.844,15:14:23.488,15:14:29.860,15:14:36.253,15:14:42.309,15:14:48.018,15:14:53.602,15:14:59.712,15:15:05.412,15:15:11.225
只有八个半回合；
需要把人名变成三个字母，修改！

pre_ssc1 - ex_ssc1 - au_ssc1
pre_ssc2 - ex_ssc2 - au_ssc2

@ fcy
pre_fcy2 - ex_fcy1 - au_fcy2 3
pre_fcy2 - ex_fcy2 - au_fcy2



pre_lyrtest_s1/2：是鼻托垫很厚的情况下的正常信号；
pre_lyrtest_t:是鼻托垫很薄的情况下的正常信号；
验证说明还是鼻托垫比较厚的信号比较好；而且一个人的信号具有稳定性；

sh ./scripts/train.sh 

# 进度
cross是3个人的ex数据；比较完整的；


# pyfeat
scikit-learn 1.2.2












![Alt text](image.png)
# record
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_yzh1.mp4
frame start 476
Reflection detected
./dataCollection/video/pre_yzh1.mp4
frame start 353
3065 3110
imu_acc_max_left 0.40335776707025534 imu_acc_max 0.6338934845880406
imu_gyro_max_left 8.276608748511073 imu_gyro_max 8.871484430912865
au_max 4.05
3300 3345
imu_acc_max_left 0.32940870895761437 imu_acc_max 0.7074202117743957
imu_gyro_max_left 4.083372434038557 imu_gyro_max 2.4433818586207314
au_max 4.06
3534 3579
imu_acc_max_left 0.23207581870416646 imu_acc_max 0.5471436729683657
imu_gyro_max_left 1.5057342872198034 imu_gyro_max 1.4782203648537624
au_max 3.83
max 0.5957829716908245 5.517310384684087
(158815,) (158815, 18)
au_start_index au_end_index 661 11881
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_yzh2.mp4
frame start 326
Reflection detected
./dataCollection/video/pre_yzh2.mp4
frame start 350
3115 3159
imu_acc_max_left 0.6530528159365042 imu_acc_max 0.8712293316790062
imu_gyro_max_left 9.74815858630157 imu_gyro_max 11.244366854326369
au_max 4.57
3348 3393
imu_acc_max_left 0.7057019828630074 imu_acc_max 0.9188765361407611
imu_gyro_max_left 5.787246156385611 imu_gyro_max 14.187421001299438
au_max 3.63
3584 3629
imu_acc_max_left 0.7855709286949077 imu_acc_max 1.051229582808577
imu_gyro_max_left 3.254686094383811 imu_gyro_max 2.7382637077902983
au_max 4.05
max 1.0288451946412076 9.646628537815634
(157169,) (157169, 18)
au_start_index au_end_index 528 11760
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
Traceback (most recent call last):
  File "E:\Emoji_Glass\1112\deepLearning\datasetMaking\GenerateData.py", line 293, in <module>
    gd.generateData()
  File "E:\Emoji_Glass\1112\deepLearning\datasetMaking\GenerateData.py", line 283, in generateData
    self.loadImuData()
  File "E:\Emoji_Glass\1112\deepLearning\datasetMaking\GenerateData.py", line 34, in loadImuData
    self.time_data = pd.read_csv(time_path,header= None)
  File "E:\anaconda\install_files\lib\site-packages\pandas\util\_decorators.py", line 311, in wrapper
    return func(*args, **kwargs)
  File "E:\anaconda\install_files\lib\site-packages\pandas\io\parsers\readers.py", line 680, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "E:\anaconda\install_files\lib\site-packages\pandas\io\parsers\readers.py", line 575, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "E:\anaconda\install_files\lib\site-packages\pandas\io\parsers\readers.py", line 933, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "E:\anaconda\install_files\lib\site-packages\pandas\io\parsers\readers.py", line 1217, in _make_engine
    self.handles = get_handle(  # type: ignore[call-overload]
  File "E:\anaconda\install_files\lib\site-packages\pandas\io\common.py", line 789, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: './dataCollection/time_ex_dyw1.csv'
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_fb1.mp4
frame start 347
var of static state is too large! 7.224592754940092
var of static state is too large! 4.422927276529274
Reflection detected
./dataCollection/video/pre_fb1.mp4
frame start 426
3217 3262
imu_acc_max_left 1.1729259690802392 imu_acc_max 0.36206725819535857
imu_gyro_max_left 19.113189797071563 imu_gyro_max 2.8078021647223546
au_max 3.37
3453 3497
imu_acc_max_left 1.4491841551155251 imu_acc_max 0.46022640856363556
imu_gyro_max_left 23.991485494388105 imu_gyro_max 5.041746066700433
au_max 3.36
3687 3732
imu_acc_max_left 1.8965951485551693 imu_acc_max 0.5212479266389443
imu_gyro_max_left 18.655690063122282 imu_gyro_max 7.411229146190954
au_max 3.73
max 1.3933165497244826 18.44503192161267
(156970,) (156970, 18)
au_start_index au_end_index 549 11816
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_fb2.mp4
frame start 365
./dataCollection/video/pre_fb2.mp4
frame start 440
3238 3282
imu_acc_max_left 0.38373074019624026 imu_acc_max 0.5925411123455748
imu_gyro_max_left 12.816972294335532 imu_gyro_max 14.681116985548284
au_max 3.83
3475 3520
imu_acc_max_left 1.167502607934619 imu_acc_max 0.6087613379104888
imu_gyro_max_left 18.49609344627952 imu_gyro_max 20.142300427874993
au_max 4.11
3712 3757
imu_acc_max_left 0.7066172151178575 imu_acc_max 0.6113553015017977
imu_gyro_max_left 13.043848118205574 imu_gyro_max 16.652295585363785
au_max 4.04
max 0.8444275805409114 19.942714553549436
(156657,) (156657, 18)
au_start_index au_end_index 599 11843
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_mrq1.mp4
frame start 305
var of static state is too large! 2.4231747588006542
./dataCollection/video/pre_mrq3.mp4
frame start 394
3247 3292
imu_acc_max_left 0.8091725267671174 imu_acc_max 0.4762865917334041
imu_gyro_max_left 11.43989846601955 imu_gyro_max 3.672746172302945
au_max 3.97
3484 3529
imu_acc_max_left 1.158172645052172 imu_acc_max 0.5461905651958384
imu_gyro_max_left 19.287469350403526 imu_gyro_max 6.406656541073203
au_max 4.12
3719 3764
imu_acc_max_left 1.383708231707151 imu_acc_max 0.5576071629078568
imu_gyro_max_left 13.722492492968009 imu_gyro_max 7.063151159286321
au_max 4.04
max 1.0149976935056702 12.656759265863363
(154786,) (154786, 18)
au_start_index au_end_index 463 11691
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_mrq2.mp4
frame start 328
./dataCollection/video/pre_mrq4.mp4
frame start 351
3045 3090
imu_acc_max_left 2.400346623545071 imu_acc_max 0.39309119226191785
imu_gyro_max_left 26.48602657190179 imu_gyro_max 7.545035057174766
au_max 4.43
3280 3325
imu_acc_max_left 2.5951393300822847 imu_acc_max 0.3250621706882936
imu_gyro_max_left 24.917753976635428 imu_gyro_max 14.630660744915652
au_max 3.94
3517 3561
imu_acc_max_left 2.1304560866591338 imu_acc_max 0.2993166545946097
imu_gyro_max_left 26.60590489339234 imu_gyro_max 16.24886455577447
au_max 3.56
max 1.7118843307653027 24.797900432423916
(154939,) (154939, 18)
au_start_index au_end_index 439 11661
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_lls1.mp4
frame start 542
Reflection detected
./dataCollection/video/pre_lls1.mp4
frame start 209
3016 3060
imu_acc_max_left 0.7229146029783081 imu_acc_max 0.35383787270527534
imu_gyro_max_left 9.77168107596211 imu_gyro_max 4.601780636739069
au_max 4.45
3249 3294
imu_acc_max_left 0.6912088543956282 imu_acc_max 0.38910794749539873
imu_gyro_max_left 2.321105350631919 imu_gyro_max 1.1348062140789237
au_max 4.48
3484 3529
imu_acc_max_left 0.7583406476604219 imu_acc_max 0.38157077741726786
imu_gyro_max_left 4.659057206203492 imu_gyro_max 5.405522528469838
au_max 4.56
max 0.610907972852 5.173787421814772
(150147,) (150147, 18)
au_start_index au_end_index 653 11873
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_lls2.mp4
frame start 471
Reflection detected
./dataCollection/video/pre_lls2.mp4
Traceback (most recent call last):
  File "E:\Emoji_Glass\1112\deepLearning\datasetMaking\GenerateData.py", line 293, in <module>
    gd.generateData()
  File "E:\Emoji_Glass\1112\deepLearning\datasetMaking\GenerateData.py", line 286, in generateData
    self.acc_maximum, self.gyro_maximum = self.calculate_maximum()
  File "E:\Emoji_Glass\1112\deepLearning\datasetMaking\GenerateData.py", line 100, in calculate_maximum
    first_frame = self.video_capture(video_path = calibra_video_path)
  File "E:\Emoji_Glass\1112\deepLearning\datasetMaking\GenerateData.py", line 217, in video_capture
    cv2.namedWindow("Capture")
KeyboardInterrupt
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_lls2.mp4
frame start 440
Reflection detected
./dataCollection/video/pre_lls2.mp4
frame start 244
2871 2916
imu_acc_max_left 0.651910454470306 imu_acc_max 1.1832616899489343
imu_gyro_max_left 6.532751498217667 imu_gyro_max 9.213700391149303
au_max 4.59
3105 3150
imu_acc_max_left 0.6513172556209221 imu_acc_max 1.3475214343456134
imu_gyro_max_left 6.869375740242602 imu_gyro_max 4.698305101994746
au_max 4.63
3340 3384
imu_acc_max_left 0.29136047686285915 imu_acc_max 1.3252680841117404
imu_gyro_max_left 3.063812055728855 imu_gyro_max 2.820000195293164
au_max 4.71
max 0.9789729160768355 5.981863139157368
(142832,) (142832, 18)
au_start_index au_end_index 664 11874
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_dll2.mp4
frame start 310
Reflection detected
./dataCollection/video/pre_dll2.mp4
frame start 341
3331 3376
imu_acc_max_left 0.6047844900046295 imu_acc_max 0.6463566915062879
imu_gyro_max_left 7.716630609353285 imu_gyro_max 4.160134303709907
au_max 4.34
3567 3611
imu_acc_max_left 1.5713982177541566 imu_acc_max 1.037544560528355
imu_gyro_max_left 43.84402571641892 imu_gyro_max 34.24764543263326
au_max 3.77
3801 3846
imu_acc_max_left 1.2726706120668425 imu_acc_max 1.186240599915018
imu_gyro_max_left 23.858288763811725 imu_gyro_max 41.231612732870374
au_max 3.43
max 1.414326979703253 35.356000475786836
(150561,) (150561, 18)
au_start_index au_end_index 407 11634
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_dll1.mp4
frame start 319
Reflection detected
./dataCollection/video/pre_dll1.mp4
frame start 308
3009 3054
imu_acc_max_left 0.7578597180869433 imu_acc_max 0.7722810809355247
imu_gyro_max_left 7.265596960173401 imu_gyro_max 6.835619866815281
au_max 3.9
3244 3289
imu_acc_max_left 0.6993776985783233 imu_acc_max 0.7948241568853079
imu_gyro_max_left 8.888845820103548 imu_gyro_max 9.749773714160963
au_max 3.86
3479 3524
imu_acc_max_left 1.1660985570202531 imu_acc_max 0.8072718973081038
imu_gyro_max_left 9.651020358593929 imu_gyro_max 13.728137116492958
au_max 3.75
max 1.0880623683844075 12.232330438410429
(152514,) (152514, 18)
au_start_index au_end_index 457 11663
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_nxq2.mp4
frame start 303
./dataCollection/video/pre_nxq4.mp4
frame start 229
3034 3079
imu_acc_max_left 0.4919907775541001 imu_acc_max 0.4661359194371916
imu_gyro_max_left 3.977461759446509 imu_gyro_max 8.017596093162055
au_max 2.91
3269 3314
imu_acc_max_left 0.5069457862137585 imu_acc_max 0.4027275349992857
imu_gyro_max_left 3.8354702343764373 imu_gyro_max 5.476922550903422
au_max 2.93
3503 3548
imu_acc_max_left 0.5506484692330162 imu_acc_max 0.46445076717552736
imu_gyro_max_left 8.223053721173558 imu_gyro_max 7.114839951225403
au_max 2.87
max 0.827845820434105 10.537098236584441
(149009,) (149009, 18)
au_start_index au_end_index 531 11748
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_nxq1.mp4
frame start 343
var of static state is too large! 31.94661291273635
Reflection detected
./dataCollection/video/pre_nxq3.mp4
frame start 255
2884 2929
imu_acc_max_left 0.750270693038825 imu_acc_max 0.2585986391031723
imu_gyro_max_left 15.5749805773503 imu_gyro_max 5.265379303024732
au_max 2.73
3119 3164
imu_acc_max_left 0.6838492951399875 imu_acc_max 0.19574032642765316
imu_gyro_max_left 10.068184498413116 imu_gyro_max 12.359530193256177
au_max 3.01
3354 3399
imu_acc_max_left 0.7788301535815445 imu_acc_max 0.2944091909660808
imu_gyro_max_left 9.740323813289143 imu_gyro_max 17.43754484647803
au_max 3.02
max 0.8476241149636733 20.070161433408938
(145809,) (145809, 18)
au_start_index au_end_index 542 11783
(base) 
kidominox@LAPTOP-953Q51TB MINGW64 /e/Emoji_Glass/1112
$ python ./deepLearning/datasetMaking/GenerateData.py 
./dataCollection/video/ex_cyx1.mp4
frame start 250
var of static state is too large! 2.3299968355156335
./dataCollection/video/pre_cyx3.mp4
frame start 321
3034 3079
imu_acc_max_left 0.5366856277525232 imu_acc_max 0.6749658111527921
imu_gyro_max_left 2.4200366942874076 imu_gyro_max 4.700102018119418
au_max 4.59
3268 3313
imu_acc_max_left 0.5183008422698885 imu_acc_max 0.5847878967373207
imu_gyro_max_left 4.178487926209942 imu_gyro_max 6.047227612917937
au_max 4.73
3500 3545
imu_acc_max_left 0.49723836784709935 imu_acc_max 0.667097535557017
imu_gyro_max_left 2.6832001304295083 imu_gyro_max 1.6767955308086142
au_max 4.97
max 0.6095502631896181 3.825313398450309
(146551,) (146551, 18)
au_start_index au_end_index 35