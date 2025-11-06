import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QGridLayout,
     QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog)
from PyQt5.QtGui import QPixmap

import argparse
import os
import json
import operator
import math
import re

from os import path

import numpy as np
import librosa
import matplotlib.pyplot as plt

def process_pic(xx, x_sq, red_line, other_line, buttom_index, MDI, time_index):
    #plt.subplot(211)
    if(len(time_index[xx]) > len(x_sq[xx])):
        temp_x = time_index[xx][:len(x_sq[xx])]
    elif(len(time_index[xx]) < len(x_sq[xx])):
        sys.exit('plot x<y')
        temp_y = x_sq[xx][:len(time_index[xx])]
    else:
        temp_x = time_index[xx]
    plt.plot(temp_x, x_sq[xx], label='x_sq')
    plt.plot(temp_x, red_line[xx], label='red_line')
    plt.plot(temp_x, other_line[xx], label='other_line')
    plt.plot(temp_x, lab_sq[xx], label='lab')
    
    #plt.legend()
    #plt.grid()
    #plt.subplot(212)
    #plt.plot(t, target_sq[xx], label='target_sq')
    #plt.plot(t, other_line)
    plt.legend()
    plt.grid()
    plt.savefig("app_show.png")
    plt.close()
    plt.clf()
    
    if(len(x_sq[xx])<256):
        x_sq_2 = np.zeros(256)
        x_sq_2[:len(x_sq[xx])] = x_sq[xx]
    else:
        x_sq_2 = x_sq[xx]
    plt.specgram(x_sq_2[:len(x_sq[xx])],Fs=fs)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig("app_spectrogram.png")
    plt.close()
    plt.clf()
    #print('app_show.png')
    
def read_lab(labf):
    starts, stops, labs, words = [], [], [], []
    file = open(labf,"r")
    sents = file.readlines()
    C = []
    D = []
    begin = 0
    end = len(sents)
    begin_check = 0
    for i in range(begin, end):
        #print(sents[i])
        x2 = re.split("\\n", sents[i])
        #print(x2)
        
        x = re.split(r' ', x2[0])
        #print(x)
        x1 = []
        for j in range(len(x)):
            if(x[j] != ""):
                x1.append(x[j])
        #print(x1)
        if(begin_check == 0):
            A = int(x1[0])
            B = int(x1[1])
            C = []
            D = []
            C.append(x1[2])
            if(len(x) == 4):
                D.append(x1[3])
            else:
                D.append("")
            begin_check = 1
        else:
            temp1 = int(x1[0])
            temp2 = int(x1[1])
            A = np.r_[A, temp1]
            B = np.r_[B, temp2]
            C.append(x1[2])
            if(len(x) == 4):
                D.append(x1[3])
            else:
                D.append("")
        x1.clear()
        #os.system("pause")
        '''
        if(K == 5): # K = 3580
            sys.exit('pause')
        '''
    
    return A, B, C, D

def lab_sq_process(stops_al, x_sq):
    global lab_sq, lab_x, lab_sq_fixpoint
    lab_x = np.zeros(len(x))
    for kk in range(len(stops_al)):
        stop_point = int(round(stops_al[kk]/10000000*fs))
        lab_x[stop_point] = -1
    for kk in range(len(sq_start_point)):
        lab_sq_ = np.zeros(len(x_sq[kk]))
        lab_sq_ = lab_x[sq_start_point[kk]:sq_stop_point[kk]]
        min_a = np.min(x_sq[kk])
        for mm in range(len(lab_sq_)):
            if(lab_sq_[mm] == -1):
                lab_sq_[mm] = min_a
        lab_sq.append(lab_sq_)
        '''
        center = int(round((sq_stop_point[kk]-sq_start_point[kk])/2 + sq_start_point[kk]))
        lab_sq_2 = lab_sq_.copy()
        for mm in range(len(lab_sq_2)):
            if(mm >= center-10 and mm <= center+10):
                pass
            else:
                lab_sq_2[mm] = 0
        lab_sq_fixpoint.append(lab_sq_2)
        '''
        
def read_data(wavf, d_KLf, labf):
    # read data
    #wavf = os.path.join(spath + file_path_al[i] + '.wav')
    #d_KLf = os.path.join(tpath + file_path_al[i] + '.txt')
    # read wav
    global x, fs, starts_al, stops_al, labs_al, words_al
    global space, space_hf, d_KL
    x, fs = librosa.load(wavf, sr=None) #fs, x = wavfile.read(wavf)
    x = np.array(x, dtype=np.float)
    
    space = int(space_s * fs)
    space_hf = int(space / 2)
    # read d_KL
    f_txt = open(d_KLf,'r')
    sents_A = f_txt.readlines()
    d_KL = []
    for kk in range(len(sents_A)):
        temp_sents_A = re.split("\\n", sents_A[kk])
        if(str(temp_sents_A[0]) != ''):
            d_KL.append(temp_sents_A[0])
    f_txt.close()
    #read lab
    starts_al, stops_al, labs_al, words_al = read_lab(labf)

def max_index_for_sq(target_sq_, target_peak_1_):
    max_dKL = -100
    center = int(len(target_sq_)/2)
    #center_side = 0.03 * fs
    center_side = int(len(target_sq_)/4)
    for mm in range(len(target_peak_1_)): # target_peak_1 is xx-th sq index
        if(target_peak_1_[mm]>=center-center_side and target_peak_1_[mm]<=center+center_side):
            if(max_dKL <= float(target_sq_[target_peak_1_[mm]])):
                max_dKL = float(target_sq_[target_peak_1_[mm]])
                max_dKL_index = target_peak_1_[mm]
    if(max_dKL_index > 0):
        return max_dKL_index
    else:
        sys.exit('max_dKL_index error.')

def find_candidate(starts_al, stops_al, labs_al, words_al):
    # find uv & choose candidates
    global red_line, other_line, buttom_index, MDI, time_index
    global x_sq, target_sq, max_num_point, min_num_point
    global stop_point_sq, sq_start_point, sq_stop_point, labs_al_uv
    red_line, other_line, buttom_index, MDI, time_index = [], [], [], [], []
    x_sq, target_sq = [], []
    labs_al_uv = np.zeros(len(labs_al))
    for j in range(len(labs_al)):
        phone_table_index = 200
        for kk in range(len(phone_table)):
            if(labs_al[j] == phone_table[kk]):
                if(starts_al[j] == stops_al[j] and str(labs_al[j])=='sp'): # sp duration = 0
                    labs_al_uv[j] = 2.5
                else:
                    labs_al_uv[j] = phone_table_uv[kk]
                phone_table_index = kk
                '''
                if(phone_table_uv[kk] == 1):
                    print(j, labs_al_uv[j])
                    print(phone_table_index)
                    sys.exit('stop')
                break
                '''
        if(phone_table_index == 200):
            sys.exit('Cannot find phone_table_index')
    print('len(labs_al)')
    print(len(labs_al))
    print('len(labs_al_uv)')
    print(len(labs_al_uv))
    print(labs_al_uv[:50])
    #print(d_KL[:50])
    for kk in range(len(labs_al_uv)):
        if(kk != len(labs_al_uv)-1):
            #print(kk, kk+1)
            #print(labs_al_uv[kk], labs_al_uv[kk+1])
            #start_point = int(round(starts_al[i][kk]/10000000*fs))
            stop_point = int(round(stops_al[kk]/10000000*fs))
            #os.system("pause")
            
            if(labs_al_uv[kk] == 0): # unvoiced
                if(labs_al_uv[kk+1]==1 or (labs_al_uv[kk+1]==2.5 and kk+2<len(labs_al_uv) and labs_al_uv[kk+2]==1)):
                    if(stop_point-space_hf>=0 or stop_point+space_hf<len(labs_al)-1):
                        point_a = stop_point-space_hf
                        point_b = stop_point+space_hf
                    elif(stop_point-space_hf<0):
                        point_a = 0
                        point_b = stop_point+space_hf
                    elif(stop_point+space_hf>=len(labs_al)-1):
                        point_a = stop_point-space_hf
                        point_b = len(labs_al)
                        
                    target_sq.append(d_KL[point_a:point_b])
                    x_sq.append(x[point_a:point_b])
                    time_start = (point_a)/fs
                    time_stop = (point_b)/fs
                    time_index_ = np.arange(time_start, time_stop, 1/fs)
                    time_index.append(time_index_)
                    sq_start_point.append(point_a)
                    sq_stop_point.append(point_b)
                    stop_point_sq.append(kk)
                    
                    target_checkpoint = 1
                    
                elif(labs_al_uv[kk+1] == 2):
                    pass
                elif(labs_al_uv[kk+1] == 3):
                    pass
            elif(labs_al_uv[kk] == 1): # voiced
                if(labs_al_uv[kk+1] == 0 or (labs_al_uv[kk+1]==2.5 and kk+2<len(labs_al_uv) and labs_al_uv[kk+2]==0)):
                    if(stop_point-space_hf>=0 or stop_point+space_hf<len(labs_al)-1):
                        point_a = stop_point-space_hf
                        point_b = stop_point+space_hf
                    elif(stop_point-space_hf<0):
                        point_a = 0
                        point_b = stop_point+space_hf
                    elif(stop_point+space_hf>=len(labs_al)-1):
                        point_a = stop_point-space_hf
                        point_b = len(labs_al)
                        
                    target_sq.append(d_KL[point_a:point_b])
                    x_sq.append(x[point_a:point_b])
                    time_start = (point_a)/fs
                    time_stop = (point_b)/fs
                    time_index_ = np.arange(time_start, time_stop, 1/fs)
                    time_index.append(time_index_)
                    sq_start_point.append(point_a)
                    sq_stop_point.append(point_b)
                    stop_point_sq.append(kk)
                    
                    target_checkpoint = 1
                    
                elif(labs_al_uv[kk+1] == 2):
                    pass
                elif(labs_al_uv[kk+1] == 3):
                    pass
            elif(labs_al_uv[kk] == 2): # sp
                if(labs_al_uv[kk+1] == 0):
                    pass
                elif(labs_al_uv[kk+1] == 1):
                    pass
                elif(labs_al_uv[kk+1] == 3):
                    pass
            elif(labs_al_uv[kk] == 3): # sil
                if(labs_al_uv[kk+1] == 0):
                    pass
                elif(labs_al_uv[kk+1] == 1):
                    pass
                elif(labs_al_uv[kk+1] == 2):
                    pass
            '''
            if(target_checkpoint == 0): #source
                source_sq.append(d_KL[kk-space_hf:kk+space_hf])
            '''
            
    print('len(target_sq)')
    print(len(target_sq))
    print('len(target_sq[50])')
    print(len(target_sq[50]))
    target_peak_1 = []
    for kk in range(len(target_sq)):
        target_peak_2 = []
        for mm in range(len(target_sq[kk])):
            if(mm > 0 and mm < len(target_sq[kk])-1):
                if(target_sq[kk][mm] > target_sq[kk][mm-1] and target_sq[kk][mm] > target_sq[kk][mm+1]):
                    target_peak_2.append(mm)
        target_peak_1.append(target_peak_2)
    #target_peak.append(target_peak_1)
    
    print('len(target_sq)')
    print(len(target_sq))
    # find max peak
    print('len(x_sq)')
    print(len(x_sq))
    #xx = 50
    max_dKL_index = -100
    
    print('len(target_peak_1)')
    print(len(target_peak_1))
    
    red_line = []
    other_line = []
    buttom_index = []
    MDI = []
    d_KL_2 = d_KL.copy()
    for xx in range(len(sq_start_point)):
        target_sq_temp = d_KL_2[sq_start_point[xx]:sq_stop_point[xx]]
        red_line.append(np.zeros(len(target_sq_temp)))
        other_line.append(np.zeros(len(target_sq_temp)))
        buttom_index_ = []
        
        max_num_point = np.max(x_sq[xx])
        min_num_point = np.min(x_sq[xx])
        
        max_dKL_index = max_index_for_sq(target_sq_temp, target_peak_1[xx])
        red_line[xx][max_dKL_index] = max_num_point
        buttom_index_.append(max_dKL_index)
        d_KL_2[sq_start_point[xx]+max_dKL_index] = 0.0000001
        target_sq_temp[max_dKL_index] = 0.0000001
        MDI.append(max_dKL_index)
        
        max_dKL_index = max_index_for_sq(target_sq_temp, target_peak_1[xx])
        other_line[xx][max_dKL_index] = max_num_point
        buttom_index_.append(max_dKL_index)
        target_sq_temp[max_dKL_index] = 0.0000001
        
        max_dKL_index = max_index_for_sq(target_sq_temp, target_peak_1[xx])
        other_line[xx][max_dKL_index] = max_num_point
        buttom_index_.append(max_dKL_index)
        target_sq_temp[max_dKL_index] = 0.0000001
        
        max_dKL_index = max_index_for_sq(target_sq_temp, target_peak_1[xx])
        other_line[xx][max_dKL_index] = max_num_point
        buttom_index_.append(max_dKL_index)
        
        buttom_index_.sort()
        buttom_index.append(buttom_index_)
    return red_line, other_line, buttom_index, MDI, time_index

def cal_all(wavName, fileName, labName):
    global K
    global starts_out, stops_out, labs_out, words_out
    global push_check
    K = 0
    print('start cal_all.')
    read_data(wavName, fileName, labName)
    find_candidate(starts_al, stops_al, labs_al, words_al)
    print('finished cal_all.')
    lab_sq_process(stops_al, x_sq)
    process_pic(K, x_sq, red_line, other_line, buttom_index, MDI, time_index)
    
    # fix lab    
    starts_out = starts_al.copy()
    stops_out = stops_al.copy()
    labs_out = labs_al.copy()
    words_out = words_al.copy()
    
    push_check = np.zeros(len(x_sq))

def real_filename(wavName):
    x2 = re.split("\\n", wavName)
    x = re.split(r'/', x2[0])
    x3 = re.split(r'\.', x[-1])
    return x3[0]

class window(QWidget):
    
    def __init__(self):
        super().__init__()
        
        self.pushButton_lab = QPushButton(self)
        self.pushButton_lab.setText("lab")
        self.pushButton_lab.clicked.connect(self.openlab)
        self.pushButton_wav = QPushButton(self)
        self.pushButton_wav.setText("wav")
        self.pushButton_wav.clicked.connect(self.openwav)
        self.pushButton_dKL = QPushButton(self)
        self.pushButton_dKL.setText("d_KL")
        self.pushButton_dKL.clicked.connect(self.openfile)
        self.pushButton_test = QPushButton(self)
        self.pushButton_test.setText("test")
        self.pushButton_test.clicked.connect(self.testfun)
        self.pushButton_savefixedlab = QPushButton(self)
        self.pushButton_savefixedlab.setText("save lab")
        self.pushButton_savefixedlab.clicked.connect(self.savelab)
        
        self.pushButton = QPushButton(self)
        self.pushButton.setText("candidate 1")          #text
        self.pushButton.clicked.connect(self.choose_1)
        self.pushButton_2 = QPushButton(self)
        self.pushButton_2.setText("candidate 2")
        self.pushButton_2.clicked.connect(self.choose_2)
        self.pushButton_3 = QPushButton(self)
        self.pushButton_3.setText("candidate 3")
        self.pushButton_3.clicked.connect(self.choose_3)
        self.pushButton_4 = QPushButton(self)
        self.pushButton_4.setText("candidate 4")
        self.pushButton_4.clicked.connect(self.choose_4)
        self.pushButton_last = QPushButton(self)
        self.pushButton_last.setText("last")
        self.pushButton_last.clicked.connect(self.update_pic_last)
        self.pushButton_next = QPushButton(self)
        self.pushButton_next.setText("next")
        self.pushButton_next.clicked.connect(self.update_pic_next)
        
        self.im = QPixmap("pasted image 0.png") #要確認 Lena.png 路徑
        self.label = QLabel(self)
        self.label.setPixmap(self.im) #將 image 加入 label
        self.label.setGeometry(40,40,400,400) # 大小
        
        self.im2 = QPixmap("app_spectrogram.png") #要確認 Lena.png 路徑
        self.label_2 = QLabel(self)
        #self.label_2.setPixmap(self.im2) #將 image 加入 label
        self.label_2.setGeometry(40,40,400,400) # 大小
        
        self.setGeometry(50,50,500,500)
        self.setWindowTitle("Fix Lab")
        
        hbox2 = QHBoxLayout()   #水平佈局
        hbox2.addWidget(self.pushButton_wav)
        hbox2.addWidget(self.pushButton_lab)
        hbox2.addWidget(self.pushButton_dKL)
        hbox2.addWidget(self.pushButton_test)
        hbox2.addWidget(self.pushButton_savefixedlab)
        
        vbox1 = QVBoxLayout()  #垂直佈局
        vbox1.addWidget(self.pushButton_last)
        vbox1.addWidget(self.pushButton_next)
        
        hbox1 = QHBoxLayout()   #水平佈局
        hbox1.addWidget(self.pushButton)
        hbox1.addWidget(self.pushButton_2)
        hbox1.addWidget(self.pushButton_3)
        hbox1.addWidget(self.pushButton_4)
        hbox1.addWidget(self.pushButton_4)
        hbox1.addLayout(vbox1)

        vbox = QVBoxLayout()   #垂直佈局
        vbox.addLayout(hbox2)
        vbox.addWidget(self.label)
        vbox.addWidget(self.label_2)
        vbox.addLayout(hbox1)
        self.setLayout(vbox)
    
    def openwav(self):
        global wavName, wavName_
        wavName, filetype = QFileDialog.getOpenFileName(self,
        "選取檔案",
        "./",
        "Wav Files (*.wav)")
        print(wavName)
        wavName_ = real_filename(wavName)
        print(wavName_)
        
        if(str(wavName_)==str(labName_) and str(labName_)==str(fileName_)):
            cal_all(wavName, fileName, labName)
            self.upset_pic()
            
    def openlab(self):
        global labName, labName_
        labName, filetype = QFileDialog.getOpenFileName(self,
        "選取檔案",
        "./",
        "Lab Files (*.lab)")
        print(labName)
        labName_ = real_filename(labName)
        print(labName_)
        
        if(str(wavName_)==str(labName_) and str(labName_)==str(fileName_)):
            cal_all(wavName, fileName, labName)
            self.upset_pic()
            
    def openfile(self):
        global fileName, fileName_
        fileName, filetype = QFileDialog.getOpenFileName(self,
        "選取檔案",
        "./",
        "All Files (*);;Text Files (*.txt)")
        print(fileName)
        fileName_ = real_filename(fileName)
        print(fileName_)
        
        if(str(wavName_)==str(labName_) and str(labName_)==str(fileName_)):
            cal_all(wavName, fileName, labName)
            self.upset_pic()
            
    def savelab(self):
        global K, f_lab
        '''
        fileName2, ok2 = QFileDialog.getSaveFileName(self,
        "檔案儲存",
        "./",
        "Lab Files (*.lab)")
        print(fileName2)
        print(fileName2[-30:-4])
        '''
        spath = ''
        x2 = re.split("\\n", labName)
        x = re.split(r'/', x2[0])
        for kk in range(len(x)-2):
            spath = os.path.join(spath + x[kk] + '/')
            #print(spath)
        fix_labf = os.path.join(spath + 'lab_fix/' + labName_ + '.lab')
        #print(fix_labf)
                
        f_lab = open(fix_labf,'w')
        for kk in range(len(starts_out)):
            f_lab.write(str(starts_out[kk]))
            f_lab.write(" ")
            f_lab.write(str(stops_out[kk]))
            f_lab.write(" ")
            f_lab.write(str(labs_out[kk]))
            if(str(words_out[kk]) != ''):
                f_lab.write(" ")
                f_lab.write(str(words_out[kk]))
                f_lab.write("\n")
            else:
                f_lab.write("\n")
        #f_lab.write(".")
        f_lab.close()
        
        K = 0
    
    def choose_1(self):
        buttom_num = 1 - 1
        self.push_event(buttom_num)
        
    def choose_2(self):
        buttom_num = 2 - 1
        self.push_event(buttom_num)
        
    def choose_3(self):
        buttom_num = 3 - 1
        self.push_event(buttom_num)
        
    def choose_4(self):
        buttom_num = 4 - 1
        self.push_event(buttom_num)
    
    def push_event(self, buttom_num):
        global starts_out, stops_out, labs_out, words_out
        global push_check
        s = stop_point_sq[K]
        stops_out[s] = int(time_index[K][buttom_index[K][buttom_num]]*10000000)
        starts_out[s+1] = stops_out[s]
        push_check[K] = 1
        
        if(labs_al_uv[s+1] == 2.5): # sp duration = 0
            stops_out[s+1] = starts_out[s+1]
            starts_out[s+2] = stops_out[s+1]
        
        print('push : %d'%(int(buttom_index[K][buttom_num])))
        self.update_pic_next()
        
    def update_pic_last(self):
        global K, push_check
        # write?
        K = K - 1
        process_pic(K, x_sq, red_line, other_line, buttom_index, MDI, time_index)
        self.im = QPixmap("app_show.png")
        self.label.setPixmap(self.im)
        self.label.setGeometry(40,40,400,400)
        
        self.im2 = QPixmap("app_spectrogram.png")
        self.label_2.setPixmap(self.im2)
        self.label_2.setGeometry(40,40,400,400)
        
        QApplication.processEvents()
        
    def update_pic_next(self):
        global K, push_check
        # 
        print(buttom_index[K])
        print('answer : %d'%(int(MDI[K])))
        if(push_check[K] == 0):
            for kk in range(len(buttom_index[K])):
                if(str(MDI[K]) == str(buttom_index[K][kk])):
                    temp_num = kk
                    break
            print(temp_num)
            self.push_event(temp_num)
            push_check[K] = 0
        # 
        elif(push_check[K] == 1):
            K = K + 1
            process_pic(K, x_sq, red_line, other_line, buttom_index, MDI, time_index)
            self.im = QPixmap("app_show.png")
            self.label.setPixmap(self.im)
            self.label.setGeometry(40,40,400,400)
        
            self.im2 = QPixmap("app_spectrogram.png")
            self.label_2.setPixmap(self.im2)
            self.label_2.setGeometry(40,40,400,400)
        
            QApplication.processEvents()
    
    def upset_pic(self):
        self.im = QPixmap("app_show.png")
        self.label.setPixmap(self.im)
        self.label.setGeometry(40,40,400,400)
        
        self.im2 = QPixmap("app_spectrogram.png")
        self.label_2.setPixmap(self.im2)
        self.label_2.setGeometry(40,40,400,400)
        QApplication.processEvents()
    
    def testfun(self):
        global wavName, labName, fileName
        global wavName_, labName_, fileName_
        
        wavName = './UDN_raw/Rickie/treebank_normal/speech/Rickie-treebank_normal-001.wav'
        labName = './UDN_raw/Rickie/treebank_normal/lab/Rickie-treebank_normal-001.lab'
        fileName = './UDN_raw/Rickie/treebank_normal/d_KL/Rickie-treebank_normal-001.txt'
        
        wavName_ = real_filename(wavName)
        labName_ = real_filename(labName)
        fileName_ = real_filename(fileName)
        
        if(not os.path.exists(wavName)):
            sys.exit('Cannot find %s'%wavName)
        
        if(str(wavName_)==str(labName_) and str(labName_)==str(fileName_)):
            cal_all(wavName, fileName, labName)
            self.im = QPixmap("app_show.png")
            self.label.setPixmap(self.im)
            self.label.setGeometry(40,40,400,400)
            
            self.im2 = QPixmap("app_spectrogram.png")
            self.label_2.setPixmap(self.im2)
            self.label_2.setGeometry(40,40,400,400)
            
            QApplication.processEvents()
            
space_s = 0.3 # left space_s/2 s right space_s/2 s
K = 0 # peak index in a file
space = 0
space_hf = 0
d_KL = []
x_sq = []
target_sq = []
lab_sq, lab_sq_fixpoint = [], []
sq_start_point, sq_stop_point = [], []
stop_point_sq = []
starts_al, stops_al, labs_al, words_al = [], [], [], []
red_line, other_line, buttom_index, MDI, time_index = [], [], [], [], []
wavName_ = ''
labName_ = ''
fileName_ = ''
wavName = ''
labName = ''
fileName = ''
if __name__ == '__main__':
    # all parameter
    
    
    # read phoneme table
    phone_table = []
    phone_table_file = open("phoneme_table.txt","r")
    sents_A = phone_table_file.readlines()
    for i in range(len(sents_A)):
        temp_sents_A = re.split("\\n", sents_A[i])
        temp_sents_B = re.split(r' ', temp_sents_A[0])
        phone_table.append(temp_sents_B[0])
        if(temp_sents_B[1] == 'voiced'):
            if(i == 0):
                phone_table_uv = 1
            else:
                phone_table_uv = np.r_[phone_table_uv, 1]
        elif(temp_sents_B[1] == 'unvoiced'):
            if(i == 0):
                phone_table_uv = 0
            else:
                phone_table_uv = np.r_[phone_table_uv, 0]
    
    phone_table_file.close()
    phone_table.append('sp')
    phone_table.append('sil')
    phone_table_uv = np.r_[phone_table_uv, 2]
    phone_table_uv = np.r_[phone_table_uv, 3]
    
    # app main
    app = QApplication(sys.argv)
    ex = window()
    ex.show() #將show 寫到外面
    sys.exit(app.exec_())
