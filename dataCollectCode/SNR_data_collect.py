import pyttsx3
import csv
import time
import numpy as np
from datetime import datetime
import sys
import os
from skimage import io
import matplotlib.pyplot as plt
import pythoncom
import PyHook3 as pyHook
import mouse
"""
@Description :  实验设计-指导实验者采集数据 数据能及时地保存
@Author      :  kidominox 
@Time        :   2023/07/27 00:08:26
"""


epoch = 10
title_texts = "准备开始实验"
calibration_tests = "开始5秒静止校准"
calibration_end_tests = "结束5秒校准"
ex_pre_texts = ["下一回合是静止","下一回合是开心","下一回合是难过","下一回合是害怕","下一回合是生气","下一回合是惊喜","下一回合是厌恶"]
ex_texts = ["静止","开心","难过","害怕","生气","惊喜","厌恶"]

end_texts = "结束动作"
start_record = False
AU_pre_texts = ["下一回合内部眉毛抬起AU1","下一回合是外部眉毛抬起AU2","下一回合是皱眉AU4","下一回合是抬上眼皮AU5","下一回合是抬起脸颊AU6",\
                "下一回合是眼睛收缩AU7","下一回合是收缩抬起鼻子AU9","下一回合是抬起上嘴唇AU10","下一回合是上扬嘴角AU12","下一回合是形成酒窝AU14",\
                    "下一回合是嘴角垂直拉动AU15","下一回合是挤动下唇向上拉动AU17","下一回合是嘴唇向后拉扯AU20","下一回合是收紧双唇成一线AU24",\
                        "下一回合是微微张开嘴巴AU25","下一回合是张大嘴巴AU26","下一回合是双目眨眼AU45"]
AU_texts = ["眉毛抬起","皱眉","瞪大眼睛",\
                "眼睛收缩","收缩抬起鼻子","抬起脸颊上扬嘴角",\
                    "挤动下唇向上拉动","嘴唇向后拉扯","收紧双唇成一线",\
                        "微微张开嘴巴","张大嘴巴"]

pre_texts = ["下一回合是静止","下一回合是开心","下一回合是皱眉","下一回合是张大嘴巴"]
texts = ["静止","开心","皱眉","张大嘴巴"]

def write_csv(data_row):
    path = str(csv_path)#同目录下的新文件名                            
    with open(path,mode='a',newline = '',encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

# 文件的表头
def st_write_csv(data_row):                          
    with open(index_csv_path,mode='a',newline = '',encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

        
def pre_collect():
    # collect the data of pre emotion of the experiment
    global start_record
    engine = pyttsx3.init()
    engine.say(title_texts)
    engine.runAndWait()


    # 每个表情5分钟
    # shuffled_indices = np.random.permutation(len(ex_texts))
    index_list = []
    for j in range(len(texts)):
        save_list = []
        index_list.append(j)
        engine.say(pre_texts[j])
        engine.runAndWait()
        time.sleep(2)
        
        for i in range(3):
            # 开始采集
            engine.say(texts[j])
            engine.runAndWait()
             # 采集时间
            dt = datetime.today()
            save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
            time.sleep(1.5)
            dt = datetime.today()
            save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
            engine.say(end_texts)
            engine.runAndWait()
            # 休息
            time.sleep(3)
        write_csv(save_list)
    st_write_csv(index_list)

def exCollect():
    # 声音播报表情，然后等待四秒，继续播报下一个表情；表情每一个epoch会被打乱顺序
    engine = pyttsx3.init()
    engine.say(title_texts)
    engine.runAndWait()


    for i in range(epoch):
        shuffled_indices = np.random.permutation(len(ex_texts))
        index_list = []

        save_list = []
        for j in range(len(shuffled_indices)):
            
            index_list.append(shuffled_indices[j])
            # engine.say(ex_pre_texts[shuffled_indices[j]])
            # engine.runAndWait()
            # time.sleep(2)
            # 开始采集
            engine.say(ex_texts[shuffled_indices[j]])
            engine.runAndWait()
            # 采集时间
            dt = datetime.today()
            save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
            time.sleep(4)
            # dt = datetime.today()
            # save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
            # engine.say(end_texts)
            # engine.runAndWait()
            # # 休息
            # time.sleep(2)
        write_csv(save_list)
        st_write_csv(index_list)
        print("第"+str(i+1)+"次实验结束")

def auCollect():
    # 声音播报AU，然后等待四秒，继续播报下一个AU；AU每一个epoch会被打乱顺序
    engine = pyttsx3.init()
    engine.say(title_texts)
    engine.runAndWait()

    for i in range(epoch):
        shuffled_indices = np.random.permutation(len(AU_texts))
        index_list = []
        save_list = []
        for j in range(len(shuffled_indices)):
            
            index_list.append(shuffled_indices[j])
            # engine.say(ex_pre_texts[shuffled_indices[j]])
            # engine.runAndWait()
            # time.sleep(2)
            # 开始采集
            engine.say(AU_texts[shuffled_indices[j]])
            engine.runAndWait()
            # 采集时间
            dt = datetime.today()
            save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
            time.sleep(4)
            # dt = datetime.today()
            # save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
            # engine.say(end_texts)
            # engine.runAndWait()
            # # 休息
            # time.sleep(2)
        write_csv(save_list)
        st_write_csv(index_list)
        print("第"+str(i+1)+"次实验结束")


# old version:unused
def ex_collect():
    global start_record
    engine = pyttsx3.init()
    engine.say(title_texts)
    engine.runAndWait()


    # 每个表情5分钟
    shuffled_indices = np.random.permutation(len(ex_texts))
    index_list = []
    for j in range(len(shuffled_indices)):
        save_list = []
        index_list.append(shuffled_indices[j])
        engine.say(ex_pre_texts[shuffled_indices[j]])
        engine.runAndWait()
        time.sleep(2)
        
        for i in range(epoch):
            # 开始采集
            engine.say(ex_texts[shuffled_indices[j]])
            engine.runAndWait()
            # 采集时间
            dt = datetime.today()
            save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
            time.sleep(3)
            dt = datetime.today()
            save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
            engine.say(end_texts)
            engine.runAndWait()
            # 休息
            time.sleep(2)
        write_csv(save_list)
    st_write_csv(index_list)

# old version:unused
def AU_collect():
    engine = pyttsx3.init()
    engine.say(calibration_tests)
    engine.runAndWait()
    save_list = []
    dt = datetime.today()
    save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
    
    time.sleep(5)
    dt = datetime.today()
    save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
    write_csv(save_list)
    engine.say(calibration_end_tests)
    engine.runAndWait()

    
    # 每个表情5分钟
    shuffled_indices = np.random.permutation(len(AU_texts))
    for j in range(len(shuffled_indices)):
        save_list = []
        save_list.append(shuffled_indices[j])
        engine.say(AU_pre_texts[shuffled_indices[j]])
        engine.runAndWait()
        mouse.wait(button='left')
        # # 等待键盘开始 
        # hm.HookMouse()
        # # 循环监听
        # pythoncom.PumpMessages()
        dt = datetime.today()
        save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
        engine.say(AU_texts[shuffled_indices[j]])
        engine.runAndWait()
        time.sleep(60)
        # 结束 
        dt = datetime.today()
        save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
        write_csv(save_list)
        engine.say(end_texts)
        engine.runAndWait()

# old version:unused
def natural_emotion():
    picture_text = "开始观看图片！"
    picture_text_end = "观看结束！"
    engine = pyttsx3.init()
    engine.say(calibration_tests)
    engine.runAndWait()
    save_list = []
    dt = datetime.today()
    save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
    
    time.sleep(5)
    dt = datetime.today()
    save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
    write_csv(save_list)
    engine.say(calibration_end_tests)
    engine.runAndWait()
    time.sleep(2)
    engine.say(picture_text)
    engine.runAndWait()
    save_list = []
    dt = datetime.today()
    save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
    
    for filename in os.listdir('./motion_stimulate/images'):
        img=io.imread('./motion_stimulate/images/' + filename)
        # plt.axis('off')
        io.imshow(img)
        
        plt.ion()
        plt.axis('off')
        plt.pause(0.01)
        time.sleep(5)
        plt.clf()
    dt = datetime.today()
    save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
    write_csv(save_list)
    engine.say(picture_text_end)
    engine.runAndWait()



if __name__ == "__main__":
    if len(sys.argv) > 2:
        ex_per = sys.argv[1]
        
        index_csv_path = "./dataCollection/index_"+ ex_per +".csv"
        if sys.argv[2] == 'au':
            csv_path = "./dataCollection/time_"+ ex_per +".csv"
            auCollect()
        elif sys.argv[2] == 'ex':
            csv_path = "./dataCollection/time_"+ ex_per +".csv"
            exCollect()
        elif sys.argv[2] == 'pre':
            csv_path = "./dataCollection/time_"+ ex_per +".csv"
            pre_collect()
        # elif sys.argv[2] == 'ne':
        #     csv_path = "./selectSize/ne_"+ ex_per +".csv"
        #     natural_emotion()
        else:
            print("please input au or ex.")
    else:
        print("No command-line arguments provided.")

    