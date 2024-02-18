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


epoch = 3
title_texts = "准备开始实验"
calibration_tests = "开始5秒静止校准"
calibration_end_tests = "结束5秒校准"
ex_pre_texts = ["下一回合是开心","下一回合是难过","下一回合是害怕","下一回合是生气","下一回合是惊喜","下一回合是厌恶","下一回合是轻蔑"]
ex_texts = ["开心","难过","害怕","生气","惊喜","厌恶","轻蔑"]
end_texts = "结束动作"
start_record = False
AU_pre_texts = ["下一回合内部眉毛抬起AU1","下一回合是外部眉毛抬起AU2","下一回合是皱眉AU4","下一回合是抬上眼皮AU5","下一回合是抬起脸颊AU6",\
                "下一回合是眼睛收缩AU7","下一回合是收缩抬起鼻子AU9","下一回合是抬起上嘴唇AU10","下一回合是上扬嘴角AU12","下一回合是形成酒窝AU14",\
                    "下一回合是嘴角垂直拉动AU15","下一回合是挤动下唇向上拉动AU17","下一回合是嘴唇向后拉扯AU20","下一回合是收紧双唇成一线AU24",\
                        "下一回合是微微张开嘴巴AU25","下一回合是张大嘴巴AU26","下一回合是双目眨眼AU45"]
AU_texts = ["内部眉毛抬起","外部眉毛抬起","皱眉","抬上眼皮","抬起脸颊",\
                "眼睛收缩","收缩抬起鼻子","抬起上嘴唇","上扬嘴角","形成酒窝",\
                    "嘴角垂直拉动","挤动下唇向上拉动","嘴唇向后拉扯","收紧双唇成一线",\
                        "微微张开嘴巴","张大嘴巴","双目眨眼"]
def write_csv(data_row):
    path = str(csv_path)#同目录下的新文件名                            
    with open(path,mode='a',newline = '',encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)
def st_write_csv(data_row):                          
    with open(start_csv_path,mode='a',newline = '',encoding='utf-8_sig') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

# def funcLeft(event):
#     global start_record
#     if (event.MessageName != "mouse move"):  # 因为鼠标一动就会有很多mouse move，所以把这个过滤下
#         start_record = True
#         hm.UnhookMouse()
#     return True

# def funcRight(event):
#     global start_record
#     if (event.MessageName != "mouse move"):  # 因为鼠标一动就会有很多mouse move，所以把这个过滤下
#         start_record = False
#         hm.UnhookMouse()
#     return True
def ex_collect():
    global start_record
    engine = pyttsx3.init()
    engine.say(calibration_tests)
    engine.runAndWait()
    save_list = []
    dt = datetime.today()
    save_list.append(-1)
    save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
    
    time.sleep(5)
    dt = datetime.today()
    save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
    write_csv(save_list)
    engine.say(calibration_end_tests)
    engine.runAndWait()

    
    # 每个表情5分钟
    shuffled_indices = np.random.permutation(len(ex_texts))
    for j in range(len(shuffled_indices)):
        save_list = []
        save_list.append(shuffled_indices[j])
        engine.say(ex_pre_texts[shuffled_indices[j]])
        engine.runAndWait()
        mouse.wait(button='left')
        # # 等待键盘开始 
        # hm.HookMouse()
        # try:
        #     # 循环监听鼠标事件
        #     pythoncom.PumpMessages()
        # except KeyboardInterrupt:
        #     # 用户按下 Ctrl+C 时退出程序
        #     hm.UnhookMouse()

        dt = datetime.today()
        save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
        engine.say(ex_texts[shuffled_indices[j]])
        engine.runAndWait()
        time.sleep(120)
        # 结束 
        dt = datetime.today()
        save_list.append(str(dt.hour).zfill(2) +':'+str(dt.minute).zfill(2)+':'+str(dt.second).zfill(2)+'.'+str(dt.microsecond//1000).zfill(3))
        write_csv(save_list)
        engine.say(end_texts)
        engine.runAndWait()


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
        
        start_csv_path = "./timestamp/start_"+ ex_per +".csv"
        # hm = pyHook.HookManager()
        # # 监听鼠标
        # # hm.MouseLeftDown 是将“鼠标左键按下”这一事件和func这个函数绑定，即每次鼠标左键按下都会执行func
        # # 如果希望监测鼠标中键按下则：hm.MouseMiddleDown，鼠标右键按下则：hm.MouseRightDown
        # hm.MouseLeftDown = funcLeft  # 监测鼠标左键是否按下
        # hm.MouseRightDown = funcRight  # 监测鼠标左键是否按下
        if sys.argv[2] == 'au':
            csv_path = "./timestamp/au_"+ ex_per +".csv"
            AU_collect()
        elif sys.argv[2] == 'ex':
            csv_path = "./timestamp/ex_"+ ex_per +".csv"
            ex_collect()
        elif sys.argv[2] == 'au2':
            csv_path = "./timestamp/ex2_"+ ex_per +".csv"
            AU_collect()
        elif sys.argv[2] == 'ne':
            csv_path = "./timestamp/ne_"+ ex_per +".csv"
            natural_emotion()
        else:
            print("please input au or.")
    else:
        print("No command-line arguments provided.")

    