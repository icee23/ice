# -*- coding: utf-8 -*-

"""
dtw_BLSTM.py None/diff phone/others

"""

import os
import sys
import json
import numpy as np 
import torch as th
import torch.nn as nn 
import torch.utils.data as data
import torch.optim as optim
import argparse as ap
import operator
from sklearn import preprocessing
import h5py

import scipy.sparse

from os import path

from sprocket.util import HDF5, estimate_twf, melcd
from sprocket.util import static_delta, extfrm

from src.yml import SpeakerYML, PairYML

import dnnVC
from convert_lib import read_feats
import re
import math

with open(path.join("configs","Configs_2.json")) as configs_file:
    configs = json.load(configs_file)

max_epoch = 15
#max_epoch = 10
batch_size = 1
num_layers = 2
use_cuda = configs["use_cuda"]
hidden_size = 512

mgc_order=configs["mgc_order"]
shiftms = 0.005

input_size = 40 # 40+66(22+40+2+2)+66
output_size = 40

gmmmode = configs["gmmmode"]

frame_max = 512

device = th.device("cuda" if use_cuda else "cpu")

def normalized(jnt, X_mean, X_std):
    if(ap_model == 1 and uv_model == 1):
        jnt_out = jnt[:,:mgc_order*2]
        jnt_out = preprocessing.scale(jnt_out, axis=0, with_mean=True,with_std=True)
        jnt_out = np.c_[jnt_out, jnt[:,mgc_order*2:]]
    if(ap_model == 0 and uv_model == 1):
        jnt_out = jnt[:,:mgc_order]
        jnt_out = preprocessing.scale(jnt_out, axis=0, with_mean=True,with_std=True)
        jnt_out = np.c_[jnt_out, jnt[:,mgc_order:]]
    else:
        jnt_out = jnt[:,:mgc_order]
        jnt_out = (jnt_out - X_mean) / X_std
        jnt_out = np.c_[jnt_out, jnt[:,mgc_order:]]
    return jnt_out

def align_data(org_data, tar_data, twf, remove_sp_mode, ap_model):
    if remove_sp_mode == 0 or remove_sp_mode == 2:
        jdata = np.c_[org_data[twf[0]], tar_data[twf[1]]]
    elif remove_sp_mode == 1:
        if(ap_model == 1 and uv_model == 1):
            jdata = np.c_[org_data[twf[0]], tar_data[twf[1], 80:80+68]]
            jdata = np.c_[jdata, tar_data[twf[1], :mgc_order*2]]
        else:
            jdata = np.c_[org_data[twf[0]], tar_data[twf[1], mgc_order:]]
            jdata = np.c_[jdata, tar_data[twf[1], :mgc_order]]
    '''    #need 
    elif(ap_model == 1):
        jdata = np.c_[org_data[twf[0]], tar_data[twf[1], mgc_order:mgc_order+66]]
        jdata = np.c_[jdata, tar_data[twf[1], :mgc_order*2]]
        jdata = np.c_[jdata, tar_data[twf[1], (-1*mgc_order):]]
    '''
    return jdata

def free_data(train_data_, train_data_LSTM_, train_data_in_D,
                        train_data_out_D, train_data_in, train_data_out):
    train_data_ = None
    train_data_LSTM_ = None
    train_data_in_D = None
    train_data_out_D = None
    train_data_in = None
    train_data_out = None

def read_labs(listf):
    starts, stops, labs = [], [], []
    K = 0
    with open(listf, 'r') as fp:
        for line in fp:
            f_ = line.rstrip()
            f = os.path.join(f_ + '.lab')
            file = open(f,"r")
            sents = file.readlines()
            C = []
            begin = 0
            end = len(sents)
            for i in range(begin, end):
                #print(sents[i])
                x2 = re.split("\\n", sents[i])
                x = re.split(r' ', x2[0])
                x1 = []
                for j in range(len(x)):
                    if(x[j] != ""):
                        x1.append(x[j])
                #print(x)
                if(i == begin):
                    A = float(x1[0])/10000000
                    B = float(x1[1])/10000000
                    C.append(x1[2])
                else:
                    temp1 = float(x1[0])/10000000
                    temp2 = float(x1[1])/10000000
                    A = np.r_[A, temp1]
                    B = np.r_[B, temp2]
                    C.append(x1[2])
                x1.clear()
            starts.append(A)
            stops.append(B)
            labs.append(C)
            file.close()
            A, B, C = None, None, None
            
            K = K+1
            #print(K)
    print(len(starts[1]))
    print(len(stops[1]))
    print(len(labs[1]))
    return starts, stops, labs

def remove_sp(mceps, starts, stops, labs, phone_table, error_num):
    outputmceps = []
    #print(mceps[1].shape)
    for i in range(len(starts)):
        K = 0
        mcep_t = mceps[i]
        cc = mcep_t.shape[0]
        check_put = np.zeros(cc)
        #print(mcep_t.shape)
        for k in range(starts[i].shape[0]):
            phone_index = 100
            for ii in range(len(phone_table)):
                if(str(labs[i][k]) == str(phone_table[ii])):
                    phone_index = ii
                    break
            if (phone_index == 100):
                print(str(labs[i][k]))
                print(i, k)
                sys.exit("errors! non find phone table")
            if K == 0:
                start_frame = 0
            else:
                start_frame = int(round(stops[i][k-1]/shiftms)) + 1
            stop_frame = int(round(stops[i][k]/shiftms))
            frame_len = stop_frame - start_frame + 1
            if(frame_len<0):
                print(i, k)
                print(start_frame, stop_frame, frame_len)
                sys.exit("errors! frame_len < 0")
            elif(frame_len == 0):
                #if(error_num == 4):
                    #print(i, k)
                pass
            elif(frame_len>0):
                zero_pad = np.zeros([frame_len, len(phone_table)+2])
                zero_pad[:,phone_index] = 1
                zero_pad[:,-1] = frame_len
                if frame_len == 1:
                    zero_pad_shift = 1/frame_len
                else:
                    zero_pad_shift = 1/(frame_len-1)
                for jj in range(frame_len):
                    zero_pad[jj,-2] = jj * zero_pad_shift
                aa = mcep_t[start_frame:stop_frame+1,:].shape[0]
                bb = zero_pad.shape[0]
                if (aa <= bb):
                    #print(starts[i][k], stops[i][k], labs[i][k])
                    #print(mcep_t[start_frame:stop_frame+1,:].shape)
                    #print(zero_pad.shape)
                    #print(start_frame, stop_frame, frame_len)
                    zero_pad = zero_pad[:aa,:]
                else:
                    sys.exit("errors! remove_sp: aa > bb, error_num = %d"%(error_num))
                mcep_milk = np.c_[mcep_t[start_frame:stop_frame+1,:], zero_pad]
                check_put[start_frame:stop_frame+1] = 1
                if(K==0):
                    out_mcep = mcep_milk
                else:
                    out_mcep = np.r_[out_mcep, mcep_milk]
                K = K + 1
        dd = out_mcep.shape[0]
        if(cc == dd):
            outputmceps.append(out_mcep)
        else:
            outputmceps.append(out_mcep)
            '''
            for m in range(len(check_put)):
                if(check_put[m] == 0):
                    print(m)
            print(error_num, i, cc, dd)
            if i == 2:
                sys.exit("errors! remove_sp: cc != dd, error_num = %d"%(error_num))
            '''
        check_put = None
    print(len(outputmceps))
    print(outputmceps[1].shape)
    return outputmceps

def add_ap(mceps, mcepaps):
    output = []
    for i in range(len(mceps)):
        K = 0
        mcep_A = mceps[i]
        mcepap_A = mcepaps[i]
        #print(mcep_A.shape)
        #print(mcepap_A.shape)
        if(mcep_A.shape[0] > mcepap_A.shape[0]):
            print("error")
        else:
            out_A = np.c_[mcep_A, mcepap_A[:mcep_A.shape[0],1:]]
            output.append(out_A)
        K = K + 1
    return output

def add_uv(mceps, f0s):
    temp_B = math.log(550)
    output = []
    K = 0
    for i in range(len(mceps)):
        mcep_A = mceps[i]
        f0_A = f0s[i]
        for ii in range(f0_A.shape[0]):
            if(f0_A[ii] != 0):
                f0_A[ii] = math.log(f0_A[ii])
        if(mcep_A.shape[0] > f0_A.shape[0]):
            sys.exit("errors! add_uv: mcep.shape > f0.shape")
        
        uv_book = np.zeros([mcep_A.shape[0], 2], dtype=int)
        M = 0
        switch_ = 0
        for j in range(mcep_A.shape[0]):
            if(f0_A[j] != 0 and switch_ == 0):
                uv_book[M,0] = j
                switch_ = 1
            elif(f0_A[j] == 0 and switch_ == 1):
                uv_book[M,1] = j - 1
                switch_ = 0
                M = M + 1
        uv_book = uv_book[:M,:]
        #print(uv_book.shape)
        #print(K, mcep_A.shape[0], f0_A.shape[0])
        zero_pad = np.zeros([mcep_A.shape[0], 2])
        m = 0
        j = 0
        #for j in range(mcep_A.shape[0]):
        while(j<mcep_A.shape[0]):
            if(f0_A[j] != 0): #voiced
                zero_pad[j, 0] = 1
                zero_pad[j, 1] = f0_A[j]
                #if(j<300 or j>mcep_A.shape[0]-300):
                    #print(j, zero_pad[j, :])
            else: #unvoiced
                if(j == 0 or j == mcep_A.shape[0] - 1):
                    zero_pad[j, 0] = 0
                    zero_pad[j, 1] = 0
                    #if(j<300 or j>mcep_A.shape[0]-300):
                        #print(j, zero_pad[j, :])
                else:
                    if(m == 0):
                        frame_len = uv_book[m, 0] - 1
                        aa = 0
                        bb = f0_A[uv_book[m, 0]]
                    elif(m == M):
                        frame_len = mcep_A.shape[0] - uv_book[m-1, 1] - 2
                        aa = f0_A[uv_book[m-1, 1]]
                        bb = 0
                        #print("m == M")
                    else:
                        frame_len = uv_book[m, 0] - uv_book[m-1, 1] - 1
                        aa = f0_A[uv_book[m-1, 1]]
                        bb = f0_A[uv_book[m, 0]]
                    m = m + 1
                    cc = bb - aa
                    if(frame_len<0 or frame_len==0):
                        sys.exit("errors! UV frame_len < 0 or frame_len = 0")
                    elif(frame_len>0):                        
                        frame_len = frame_len + 1
                        zero_pad_shift = 1/frame_len
                        for jj in range(1, frame_len):
                            zero_pad[j, 0] = 0
                            zero_pad[j, 1] = aa + jj * zero_pad_shift * cc
                            #if(j<300 or j>mcep_A.shape[0]-300):
                                #print(j, zero_pad[j, :])
                            j = j + 1
                        j = j - 1
            j = j + 1
		
        zero_pad[:, 1] = zero_pad[:, 1] / temp_B

        out_A = np.c_[mcep_A, zero_pad]
        output.append(out_A)
        
        K = K + 1
    return output

def save_data(jnt_mcep, train_dnndata_path, eval_dnndata_path, test_dnndata_path,
                        X_mean, X_std,
                        train_stop_, eval_stop_):
    train_dnndata = HDF5(train_dnndata_path, mode='a')
    train_dnndata.save(jnt_mcep[:train_stop_,:], ext='mcep')
    train_dnndata.save(train_len, ext='mcep_len')
    train_dnndata.save(X_mean, ext='X_mean')
    train_dnndata.save(X_std, ext='X_std')
    train_dnndata.close()

    eval_dnndata = HDF5(eval_dnndata_path, mode='a')
    eval_dnndata.save(jnt_mcep[train_stop_:eval_stop_,:], ext='mcep')
    eval_dnndata.save(eval_len, ext='mcep_len')
    eval_dnndata.close()

    test_dnndata = HDF5(test_dnndata_path, mode='a')
    test_dnndata.save(jnt_mcep[eval_stop_:,:], ext='mcep')
    test_dnndata.save(test_len, ext='mcep_len')
    test_dnndata.close()

def train(model, trn_dl, device, criterion, optimizer, type):
    model.train()
    if (type == 1): # train
        for i, (a, c) in enumerate(trn_dl):
            a, c = a.to(device), c.to(device)
            a.requires_grad_()
            optimizer.zero_grad()
            m_ = model.forward(a, None, False)

            loss = criterion(m_, th.squeeze(c))
            loss.backward()
            optimizer.step()
    elif (type == 2): # convert
        if th.cuda.is_available():
            trn_dl = trn_dl.cuda()
            for i in range(trn_dl.shape[0]):
                print(trn_dl[i,:,:].shape)
                m_ = model.forward(trn_dl[i,:,:], None, True)
                temp = m_.cpu()
                temp2 = temp.detach().numpy()
                if(i == 0):
                    y_head = temp2
                else:
                    y_head = np.r_[y_head, temp2]
    
            return y_head

# covert function ########################
def eval(model, trn_dl, val_dl, tst_dl, device, criterion):
    model.eval()
    trn_loss = 0.0
    val_loss = 0.0
    tst_loss = 0.0
    with th.no_grad():  
        #i = 0
        for i, (a, c) in enumerate( trn_dl ):
            a, c = a.to(device), c.to(device)
            m_ = model.forward(a, None, False)
            loss = criterion(m_, th.squeeze(c))
            trn_loss += loss 
        
        #i = 0
        for i,(a,c) in enumerate( val_dl ):
            a,c= a.to(device), c.to(device)
            m_ = model.forward(a, None, False)
            loss = criterion(m_, th.squeeze(c))
            val_loss += loss 

        for i,(a,c) in enumerate( tst_dl ):
            a,c= a.to(device), c.to(device)
            m_ = model.forward(a, None, False)
            #loss = -th.mean(dnnVC.mcd(m_, c))
            loss = criterion(m_, th.squeeze(c))
            tst_loss += loss 

    trn_loss /= len(trn_dl.dataset)
    val_loss /= len(val_dl.dataset)
    tst_loss /= len(tst_dl.dataset)
    return trn_loss, val_loss, tst_loss 

def transform_jnt_fortrain(array_list, train_stop, eval_stop, test_stop, frame_max, x_step, X_mean, X_std, remove_sp_mode):
    num_files = len(array_list)
    for i in range(num_files):
        if i == 0:
            a = array_list[i].shape[0]
            temp = int(a / frame_max)
            jnt = array_list[i][:temp*frame_max,:]
            jnt = np.r_[jnt, array_list[i][a-frame_max:a,:]]
        else:
            a = array_list[i].shape[0]
            temp = int(a / frame_max)
            jnt = np.r_[jnt, array_list[i][:temp*frame_max,:]]
            jnt = np.r_[jnt, array_list[i][a-frame_max:a,:]]
        if i == (train_stop-1):
            train_stop_ = jnt.shape[0]
        elif i == (eval_stop-1):
            eval_stop_ = jnt.shape[0]
        elif i == (test_stop-1):
            test_stop_ = jnt.shape[0]
    if(x_step == 0):
        jnt = normalized(jnt, X_mean, X_std)
    elif(x_step == -2):
        pass
    elif(x_step == -1): # diff data
        diff_data = jnt[:,input_size:] - jnt[:,:mgc_order] # Y - X
        #X_mean_diff = np.mean(diff_data, axis=0)
        #X_std_diff = np.std(diff_data, axis=0)
        diff_nor = np.c_[jnt[:,:input_size], diff_data] # (x, d)
        jnt_diff = normalized(diff_nor, X_mean, X_std)
        return jnt_diff
    if(remove_sp_mode == 2 and x_step == 1):
        x_file = HDF5("./data/pair/SF1-TF1/jnt/blstm3_jnt.h5", mode='r')
        X_mean = x_file.read(ext='X_mean')
        X_std = x_file.read(ext='X_std')
        x_file.close()
    elif(remove_sp_mode != 2 and uv_model == 0):
        for i in range(num_files):
            if i == 0:
                jnt_2 = array_list[i][:,:mgc_order]
            else:
                jnt_2 = np.r_[jnt_2, array_list[i][:,:mgc_order]]
            if i == (train_stop-1):
                X_mean = np.mean(jnt_2, axis=0)
                X_std = np.std(jnt_2, axis=0)
    
    return jnt, train_stop_, eval_stop_, test_stop_, X_mean, X_std

def extsddata(data, npow, power_threshold=-20):
    """Get power extract static and delta feature vector

    Paramters
    ---------
    data : array, shape (`T`, `dim`)
        Acoustic feature vector
    npow : array, shape (`T`)
        Normalized power vector
    power_threshold : float, optional,
        Power threshold
        Default set to -20

    Returns
    -------
    extsddata : array, shape (`T_new` `dim * 2`)
        Silence remove static and delta feature vector

    """

    extsddata = extfrm(data, npow,
                       power_threshold=power_threshold)
    return extsddata

def get_alignment(odata, onpow, tdata, tnpow, opow=-20, tpow=-20,
                  sd=0, cvdata=None, given_twf=None, otflag=None, remove_sp_mode=0,
                  ap_model=0, distance='melcd'):
    """Get alignment between original and target

    Paramters
    ---------
    odata : array, shape (`T`, `dim`)
        Acoustic feature vector of original
    onpow : array, shape (`T`)
        Normalized power vector of original
    tdata : array, shape (`T`, `dim`)
        Acoustic feature vector of target
    tnpow : array, shape (`T`)
        Normalized power vector of target
    opow : float, optional,
        Power threshold of original
        Default set to -20
    tpow : float, optional,
        Power threshold of target
        Default set to -20
    sd : int , optional,
        Start dimension to be used for alignment
        Default set to 0
    cvdata : array, shape (`T`, `dim`), optional,
        Converted original data
        Default set to None
    given_twf : array, shape (`T_new`, `dim * 2`), optional,
        Alignment given twf
        Default set to None
    otflag : str, optional
        Alignment into the length of specification
        'org' : alignment into original length
        'tar' : alignment into target length
        Default set to None
    distance : str,
        Distance function to be used
        Default set to 'melcd'

    Returns
    -------
    jdata : array, shape (`T_new` `dim * 2`)
        Joint static and delta feature vector
    twf : array, shape (`T_new` `dim * 2`)
        Time warping function
    mcd : float,
        Mel-cepstrum distortion between arrays

    """
    if remove_sp_mode == 0:
        oexdata = extsddata(odata[:, sd:], onpow,
                        power_threshold=opow)
        texdata = extsddata(tdata[:, sd:], tnpow,
                        power_threshold=tpow)
    elif remove_sp_mode == 1 or remove_sp_mode == 2:
        oexdata = odata[:, sd:mgc_order+1]
        texdata = tdata[:, sd:mgc_order+1]

    if cvdata is None:
        align_odata = oexdata
    else:
        if remove_sp_mode == 0:
            cvexdata = extsddata(cvdata, onpow,
                             power_threshold=opow)
        elif remove_sp_mode == 1 or remove_sp_mode == 2:
            cvexdata = cvdata
        align_odata = cvexdata

    if given_twf is None:
        twf = estimate_twf(align_odata, texdata,
                           distance=distance, otflag=otflag)
    else:
        twf = given_twf

    if remove_sp_mode == 0 or remove_sp_mode == 2:
        jdata = align_data(oexdata, texdata, twf, remove_sp_mode, ap_model)
    elif remove_sp_mode == 1:
        jdata = align_data(odata[:, sd:], tdata[:, sd:], twf, remove_sp_mode, ap_model)
    mcd = melcd(align_odata[twf[0]], texdata[twf[1]])

    return jdata, twf, mcd


def align_feature_vectors(odata, onpows, tdata, tnpows, model_sor,
                          optimizer, criterion, input_size, max_epoch, batch_size,
                          train_stop, eval_stop, test_stop, frame_max, remove_sp_mode,
                          ap_model, opow=-100, tpow=-100, itnum=3, sd=0,
                          given_twfs=None, otflag=None):
    """Get alignment to create joint feature vector

    Paramters
    ---------
    odata : list, (`num_files`)
        List of original feature vectors
    onpows : list , (`num_files`)
        List of original npows
    tdata : list, (`num_files`)
        List of target feature vectors
    tnpows : list , (`num_files`)
        List of target npows
    opow : float, optional,
        Power threshold of original
        Default set to -100
    tpow : float, optional,
        Power threshold of target
        Default set to -100
    itnum : int , optional,
        The number of iteration
        Default set to 3
    sd : int , optional,
        Start dimension of feature vector to be used for alignment
        Default set to 0
    given_twf : array, shape (`T_new` `dim * 2`)
        Use given alignment while 1st iteration
        Default set to None
    otflag : str, optional
        Alignment into the length of specification
        'org' : alignment into original length
        'tar' : alignment into target length
        Default set to None

    Returns
    -------
    jfvs : list,
        List of joint feature vectors
    twfs : list,
        List of time warping functions
    """

    num_files = len(odata)
    cvgmm, cvdata = None, None
    X_mean, X_std = None, None
    model = model_sor
    for it in range(1, itnum + 1):
        print('{}-th joint feature extraction starts.'.format(it))
        twfs, jfvs = [], []
        jfvs_T_news = []
        for i in range(num_files):
            if it == 1 and given_twfs is not None:
                gtwf = given_twfs[i]
            else:
                gtwf = None
            if it > 1:
                if remove_sp_mode == 1:
                    if(ap_model == 1 and uv_model == 0):
                        mcep_temp = np.c_[odata[i][:, sd:107], odata[i][:, mgc_order+1:107]]
                    elif(ap_model == 1 and uv_model == 1):
                        mcep_temp = np.c_[odata[i][:, sd:149], odata[i][:, 81:149]]
                    elif(ap_model == 0 and uv_model == 1):
                        mcep_temp = np.c_[odata[i][:, sd:109], odata[i][:, 41:109]]
                    else:
                        mcep_temp = np.c_[odata[i][:, sd:], odata[i][:, mgc_order+1:]]
                    mcep_ = normalized(mcep_temp, X_mean, X_std)
                elif remove_sp_mode == 0 or remove_sp_mode == 2:
                    mcep_ = normalized(odata[i][:, sd:], X_mean, X_std)
                mcep_2 = np.reshape(mcep_, (1, mcep_.shape[0], mcep_.shape[1]))
                mcep_in_D = th.from_numpy(mcep_2)
                mcep_data = mcep_in_D.float()
                cvdata = train(model, mcep_data, device, criterion, optimizer, type=2)
                
            jdata, twf, mcd = get_alignment(odata[i],
                                            onpows[i],
                                            tdata[i],
                                            tnpows[i],
                                            opow=opow,
                                            tpow=tpow,
                                            sd=sd,
                                            cvdata=cvdata,
                                            given_twf=gtwf,
                                            otflag=otflag,
                                            remove_sp_mode=remove_sp_mode,
                                            ap_model=ap_model)
            twfs.append(twf)
            jfvs.append(jdata)
            jfvs_T_news.append(jdata.shape[0])
            print('distortion [dB] for {}-th file: {}'.format(i + 1, mcd))
        jnt_data, train_stop_, eval_stop_, test_stop_, X_mean, X_std = transform_jnt_fortrain(jfvs, 
                                                                                                                 train_stop,
                                                                                                                 eval_stop,
                                                                                                                 test_stop,
                                                                                                                 frame_max,
                                                                                                                 it,
                                                                                                                 X_mean,
                                                                                                                 X_std,
                                                                                                                 remove_sp_mode)
        print(train_stop_)
        print(eval_stop_)
        print(test_stop_)
        if it != itnum:
            # train DNN, if not final iteration
            # training part #########################
            model = None
            model = model_sor
            if(ap_model == 1 and uv_model == 1):
                jnt_data = np.c_[jnt_data[:,:input_size], jnt_data[:,(-2*output_size):(-1*output_size)]]
            elif(ap_model == 1 and uv_model == 0):
                jnt_data = np.c_[jnt_data[:,:input_size], jnt_data[:,(-2*output_size):(-1*output_size)]]
            elif(ap_model == 0 and uv_model == 1):
                pass
                #jnt_data = np.c_[jnt_data[:,:input_size], jnt_data[:,(-1*output_size):]]
            train_data_ = normalized(jnt_data[:train_stop_,:], X_mean, X_std)
            print(train_data_.shape)
            train_num = int(train_data_.shape[0]/frame_max)
            train_data_LSTM_ = np.reshape(train_data_, (train_num, frame_max, input_size+output_size))
            train_data_in_D = th.from_numpy(train_data_LSTM_[:,:,:input_size]) # numpy->tensor
            train_data_out_D = th.from_numpy(train_data_LSTM_[:,:,input_size:input_size+output_size])
            train_data_in = train_data_in_D.float() #Double.tensor -> float.tensor
            train_data_out = train_data_out_D.float()
            train_data = data.TensorDataset(train_data_in, train_data_out)
            free_data(train_data_, train_data_LSTM_, train_data_in_D, train_data_out_D, train_data_in, train_data_out)
            
            eval_data_ = normalized(jnt_data[train_stop_:eval_stop_,:], X_mean, X_std)
            eval_num = int(eval_data_.shape[0]/frame_max)
            eval_data_LSTM_ = np.reshape(eval_data_, (eval_num, frame_max, input_size+output_size))
            eval_data_in_D = th.from_numpy(eval_data_LSTM_[:,:,:input_size]) # numpy->tensor
            eval_data_out_D = th.from_numpy(eval_data_LSTM_[:,:,input_size:input_size+output_size])
            eval_data_in = eval_data_in_D.float() #Double.tensor -> float.tensor
            eval_data_out = eval_data_out_D.float()
            eval_data = data.TensorDataset(eval_data_in, eval_data_out)
            free_data(eval_data_, eval_data_LSTM_, eval_data_in_D, eval_data_out_D, eval_data_in, eval_data_out)
            
            test_data_ = normalized(jnt_data[eval_stop_:,:], X_mean, X_std)
            test_num = int(test_data_.shape[0]/frame_max)
            test_data_LSTM_ = np.reshape(test_data_, (test_num, frame_max, input_size+output_size))
            test_data_in_D = th.from_numpy(test_data_LSTM_[:,:,:input_size]) # numpy->tensor
            test_data_out_D = th.from_numpy(test_data_LSTM_[:,:,input_size:input_size+output_size])
            test_data_in = test_data_in_D.float() #Double.tensor -> float.tensor
            test_data_out = test_data_out_D.float()
            test_data = data.TensorDataset(test_data_in, test_data_out)
            free_data(test_data_, test_data_LSTM_, test_data_in_D, test_data_out_D, test_data_in, test_data_out)
            
            jnt_data = None
            
            trn_dl_for_trainig = data.DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True )
            trn_dl_for_eval = data.DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False )
            val_dl = data.DataLoader(eval_data, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
            tst_dl = data.DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
            print("+==================================================================+")
            print("|  Start Training Acoustic Model of the Source : %d                |" % it)
            print("+=========+===============+===============+===============+========+")
            print("|  epoch  |   loss(trn)   |   loss(val)   |   loss(tst)   |  save  |")
            print("+=========+===============+===============+===============+========+")
            sys.stdout.flush()
            
            min_trn_loss, min_val_loss, min_tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device, criterion)
            print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(0, min_trn_loss, min_val_loss, min_tst_loss, " "))
            print("+---------+---------------+---------------+---------------+--------+")
            sys.stdout.flush()
            

            for epoch in range(1, max_epoch+1):
                train(model, trn_dl_for_trainig, device, criterion, optimizer, type=1)
                trn_loss, val_loss, tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device, criterion)
                save = False
                if val_loss < min_val_loss:
                    save = True
                    min_val_loss = val_loss
                    model_1 = model
                    #th.save(model.state_dict(), state_dict_path)
        
                if save == True:
                    print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(epoch, trn_loss, val_loss, tst_loss, "*"))
                else:
                    print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(epoch, trn_loss, val_loss, tst_loss, " "))
        
                print("+---------+---------------+---------------+---------------+--------+")
                sys.stdout.flush()
            # #################################
        it += 1
        
    return jfvs, twfs, jfvs_T_news, X_mean, X_std

argv = sys.argv

ap_model = 0
uv_model = 0
for i in range(1, len(argv)):
    if(operator.eq(argv[i], "ap")):
        ap_model = 1
    elif(operator.eq(argv[i], "uv")):
        uv_model = 1

if (operator.eq(argv[1], "None")):
    train_dnndata_path = configs["train_blstmdata_path"]
    eval_dnndata_path = configs["eval_blstmdata_path"]
    test_dnndata_path = configs["test_blstmdata_path"]
elif(operator.eq(argv[1], "diff")):
    train_dnndata_path = configs["diff_train_blstmdata_path"]
    eval_dnndata_path = configs["diff_eval_blstmdata_path"]
    test_dnndata_path = configs["diff_test_blstmdata_path"]

source_train_list = configs["source_train_list"]
source_train_lab_list = configs["source_train_lab_list"]
target_train_list = configs["target_train_list"]
target_train_lab_list = configs["target_train_lab_list"]
source_eval_list = configs["source_eval_list"]
source_eval_lab_list = configs["source_eval_lab_list"]
target_eval_list = configs["target_eval_list"]
target_eval_lab_list = configs["target_eval_lab_list"]
source_test_list = configs["source_test_list"]
source_test_lab_list = configs["source_test_lab_list"]
target_test_list = configs["target_test_list"]
target_test_lab_list = configs["target_test_lab_list"]

# read phoneme table #########################
phone_table = []
remove_sp_mode = 0
if (operator.eq(argv[2], "phone")):
    remove_sp_mode = 1
    input_size = 172
    phone_table_file = open("./data/phoneme_table.txt","r")
    sents_A = phone_table_file.readlines()
    for i in range(len(sents_A)):
        temp_sents_A = re.split("\\n", sents_A[i])
        phone_table.append(temp_sents_A[0])
    phone_table_file.close()
    print(phone_table)
    
    # change path ############################
    if (operator.eq(argv[3], "fast")):
        train_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_64fast_jnt.h5"
        eval_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_64fast_eval_jnt.h5"
        test_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_64fast_test_jnt.h5"
        
        target_train_list = "./list/fast_train.list"
        target_train_lab_list = "./list/fast_train_lab.list"
        target_eval_list = "./list/fast_eval.list"
        target_eval_lab_list = "./list/fast_eval_lab.list"
        target_test_list = "./list/fast_test.list"
        target_test_lab_list = "./list/fast_test_lab.list"
    elif (operator.eq(argv[3], "slow")):
        train_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_64slow_jnt.h5"
        eval_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_64slow_eval_jnt.h5"
        test_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_64slow_test_jnt.h5"
        
        target_train_list = "./list/slow_train.list"
        target_train_lab_list = "./list/slow_train_lab.list"
        target_eval_list = "./list/slow_eval.list"
        target_eval_lab_list = "./list/slow_eval_lab.list"
        target_test_list = "./list/slow_test.list"
        target_test_lab_list = "./list/slow_test_lab.list"
    elif (operator.eq(argv[3], "others")):
        pass
    else:
        train_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_64_jnt.h5"
        eval_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_64_eval_jnt.h5"
        test_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_64_test_jnt.h5"
elif (operator.eq(argv[2], "allin")):
    remove_sp_mode = 2
    train_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_allin_jnt.h5"
    eval_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_allin_eval_jnt.h5"
    test_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_allin_test_jnt.h5"

if(ap_model == 1 and uv_model == 1):
    input_size = 216 #40+40+66+2+66+2
if(ap_model == 0 and uv_model == 1):
    input_size = 176

'''
# read target variance #########################
tarstatspath = configs["var_path"]
tarstats_h5 = HDF5(tarstatspath, mode='r')
targvstats_var = tarstats_h5.read(ext='gv')
#targvstats_mean = tarstats_h5.read(ext='mean')
tarstats_h5.close()
#targvstats_var = 0.1 * targvstats_var
variance_ = th.from_numpy(targvstats_var)
variance = variance_.float()
# read target variance for postfilter #################
gv_post_path = configs["gv_path"]
gvpost_h5 = HDF5(gv_post_path, mode='r')
targvpost_var = gvpost_h5.read(ext='gv')
gvpost_h5.close()
'''

# mode
model = dnnVC.BiRNN( input_size, hidden_size, output_size, num_layers).to(device)
optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()) )
criterion = nn.MSELoss()
# read source and target features from HDF file
source_train_mcep = read_feats(source_train_list, ext='mcep')
source_train_npow = read_feats(source_train_list, ext='npow')
target_train_mcep = read_feats(target_train_list, ext='mcep')
target_train_npow = read_feats(target_train_list, ext='npow')

source_eval_mcep = read_feats(source_eval_list, ext='mcep')
source_eval_npow = read_feats(source_eval_list, ext='npow')
target_eval_mcep = read_feats(target_eval_list, ext='mcep')
target_eval_npow = read_feats(target_eval_list, ext='npow')

source_test_mcep = read_feats(source_test_list, ext='mcep')
source_test_npow = read_feats(source_test_list, ext='npow')
target_test_mcep = read_feats(target_test_list, ext='mcep')
target_test_npow = read_feats(target_test_list, ext='npow')

if (ap_model == 1):
    source_train_mcepap = read_feats(source_train_list, ext='mcepap')
    target_train_mcepap = read_feats(target_train_list, ext='mcepap')

    source_eval_mcepap = read_feats(source_eval_list, ext='mcepap')
    target_eval_mcepap = read_feats(target_eval_list, ext='mcepap')

    source_test_mcepap = read_feats(source_test_list, ext='mcepap')
    target_test_mcepap = read_feats(target_test_list, ext='mcepap')
    
    source_train_mcep = add_ap(source_train_mcep, source_train_mcepap)
    print("error 1")
    target_train_mcep = add_ap(target_train_mcep, target_train_mcepap)
    print("error 2")
    source_eval_mcep = add_ap(source_eval_mcep, source_eval_mcepap)
    print("error 3")
    target_eval_mcep = add_ap(target_eval_mcep, target_eval_mcepap)
    print("error 4")
    source_test_mcep = add_ap(source_test_mcep, source_test_mcepap)
    print("error 5")
    target_test_mcep = add_ap(target_test_mcep, target_test_mcepap)
    print("error 6")
    
    free_data(source_train_mcepap, target_train_mcepap, source_eval_mcepap,
                    target_eval_mcepap, source_test_mcepap, target_test_mcepap)
    print("ap finished")

if (operator.eq(argv[2], "phone")):
    source_train_starts, source_train_stops, source_train_lab = read_labs(source_train_lab_list)
    source_eval_starts, source_eval_stops, source_eval_lab = read_labs(source_eval_lab_list)
    source_test_starts, source_test_stops, source_test_lab = read_labs(source_test_lab_list)
    error_num = 1
    source_train_mcep = remove_sp(source_train_mcep, source_train_starts, source_train_stops,
                                                      source_train_lab, phone_table, error_num)
    error_num = 2
    source_eval_mcep = remove_sp(source_eval_mcep, source_eval_starts, source_eval_stops,
                                                      source_eval_lab, phone_table, error_num)
    error_num = 3
    source_test_mcep = remove_sp(source_test_mcep, source_test_starts, source_test_stops,
                                                      source_test_lab, phone_table, error_num)

    target_train_starts, target_train_stops, target_train_lab = read_labs(target_train_lab_list)
    target_eval_starts, target_eval_stops, target_eval_lab = read_labs(target_eval_lab_list)
    target_test_starts, target_test_stops, target_test_lab = read_labs(target_test_lab_list)
    error_num = 4
    target_train_mcep = remove_sp(target_train_mcep, target_train_starts, target_train_stops,
                                                      target_train_lab, phone_table, error_num)
    error_num = 5
    target_eval_mcep = remove_sp(target_eval_mcep, target_eval_starts, target_eval_stops,
                                                      target_eval_lab, phone_table, error_num)
    error_num = 6
    target_test_mcep = remove_sp(target_test_mcep, target_test_starts, target_test_stops,
                                                      target_test_lab, phone_table, error_num)
    free_data(source_train_starts, source_train_stops, source_train_lab,
                    source_eval_starts, source_eval_stops, source_eval_lab)
    print("remove_sp finished")
else:
    pass

if (uv_model == 1):
    source_train_f0 = read_feats(source_train_list, ext='f0')
    target_train_f0 = read_feats(target_train_list, ext='f0')

    source_eval_f0 = read_feats(source_eval_list, ext='f0')
    target_eval_f0 = read_feats(target_eval_list, ext='f0')

    source_test_f0 = read_feats(source_test_list, ext='f0')
    target_test_f0 = read_feats(target_test_list, ext='f0')
    
    source_train_mcep = add_uv(source_train_mcep, source_train_f0)
    target_train_mcep = add_uv(target_train_mcep, target_train_f0)
    source_eval_mcep = add_uv(source_eval_mcep, source_eval_f0)
    target_eval_mcep = add_uv(target_eval_mcep, target_eval_f0)
    source_test_mcep = add_uv(source_test_mcep, source_test_f0)
    target_test_mcep = add_uv(target_test_mcep, target_test_f0)
    
    free_data(source_train_f0, target_train_f0, source_eval_f0, target_eval_f0,
                    source_test_f0, target_test_f0)
    print("uv finished")

print(source_train_mcep[0].shape)
org_mceps, tar_mceps = [], []
for i in range(len(source_train_mcep)):
    org_mceps.append(source_train_mcep[i])
for i in range(len(source_eval_mcep)):
    org_mceps.append(source_eval_mcep[i])
for i in range(len(source_test_mcep)):
    org_mceps.append(source_test_mcep[i])

for i in range(len(target_train_mcep)):
    tar_mceps.append(target_train_mcep[i])
for i in range(len(target_eval_mcep)):
    tar_mceps.append(target_eval_mcep[i])
for i in range(len(target_test_mcep)):
    tar_mceps.append(target_test_mcep[i])

org_npows, tar_npows = [], []
for i in range(len(source_train_npow)):
    org_npows.append(source_train_npow[i])
for i in range(len(source_eval_npow)):
    org_npows.append(source_eval_npow[i])
for i in range(len(source_test_npow)):
    org_npows.append(source_test_npow[i])

for i in range(len(target_train_npow)):
    tar_npows.append(target_train_npow[i])
for i in range(len(target_eval_npow)):
    tar_npows.append(target_eval_npow[i])
for i in range(len(target_test_npow)):
    tar_npows.append(target_test_npow[i])

print(len(source_train_mcep))
print(len(source_eval_mcep))
print(len(source_test_mcep))
train_stop = len(source_train_mcep)
eval_stop = len(source_eval_mcep) + train_stop
test_stop = len(source_test_mcep) + eval_stop
print(train_stop)
print(eval_stop)
print(test_stop)

assert len(org_mceps) == len(tar_mceps)
assert len(org_npows) == len(tar_npows)
assert len(org_mceps) == len(org_npows)

# dtw between original and target w/o 0th and silence ############
print('## Alignment mcep w/o 0-th and silence ##')
jmceps, twfs, jfvs_T_news, X_mean, X_std = align_feature_vectors(org_mceps,
                                                                    org_npows,
                                                                    tar_mceps,
                                                                    tar_npows,
                                                                    model, optimizer, criterion, input_size,
                                                                    max_epoch,batch_size,
                                                                    train_stop,eval_stop,test_stop, frame_max,
                                                                    remove_sp_mode, ap_model,
                                                                    opow=configs["power_threshold"],
                                                                    tpow=configs["power_threshold"],
                                                                    itnum=configs["n_iter"],
                                                                    sd=1)
if(ap_model == 0 and uv_model == 0):
    jnt_mcep, train_stop_, eval_stop_, test_stop_, X_mean, X_std = transform_jnt_fortrain(jmceps, train_stop, eval_stop, test_stop, frame_max, 0, X_mean, X_std, remove_sp_mode)
elif(ap_model == 0 and uv_model == 1):
    jnt_mcep, train_stop_, eval_stop_, test_stop_, X_mean, X_std = transform_jnt_fortrain(jmceps, train_stop, eval_stop, test_stop, frame_max, -2, X_mean, X_std, remove_sp_mode)
elif(ap_model == 1):
    jnt_mcep, train_stop_, eval_stop_, test_stop_, X_mean, X_std = transform_jnt_fortrain(jmceps, train_stop, eval_stop, test_stop, frame_max, -2, X_mean, X_std, remove_sp_mode)
if (operator.eq(argv[1], "diff")):
    jnt_diff_mcep = transform_jnt_fortrain(jmceps, train_stop, eval_stop, test_stop, frame_max, -1, X_mean, X_std, remove_sp_mode)
num_len = len(jfvs_T_news)
train_len = np.zeros(len(source_train_mcep), int)
eval_len = np.zeros(len(source_eval_mcep), int)
test_len = np.zeros(len(source_test_mcep), int)
j = 0
k = 0
for i in range(num_len):
    if (i < train_stop):
        train_len[i] = jfvs_T_news[i]
    elif(i >= train_stop and i < eval_stop):
        eval_len[j] = jfvs_T_news[i]
        j = j + 1
    elif(i >= eval_stop and i < test_stop):
        test_len[k] = jfvs_T_news[i]
        k = k + 1
# save joint feature vectors
if(ap_model == 1 or uv_model == 1):
    pass
elif (operator.eq(argv[1], "None")):
    train_dnndata = HDF5(train_dnndata_path, mode='a')
    train_dnndata.save(jnt_mcep[:train_stop_,:], ext='mcep')
    train_dnndata.save(train_len, ext='mcep_len')
    train_dnndata.save(X_mean, ext='X_mean')
    train_dnndata.save(X_std, ext='X_std')
    train_dnndata.close()

    eval_dnndata = HDF5(eval_dnndata_path, mode='a')
    eval_dnndata.save(jnt_mcep[train_stop_:eval_stop_,:], ext='mcep')
    eval_dnndata.save(eval_len, ext='mcep_len')
    eval_dnndata.close()

    test_dnndata = HDF5(test_dnndata_path, mode='a')
    test_dnndata.save(jnt_mcep[eval_stop_:,:], ext='mcep')
    test_dnndata.save(test_len, ext='mcep_len')
    test_dnndata.close()
    
elif (operator.eq(argv[1], "diff")):
    train_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_diff_64_jnt.h5"
    eval_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_diff_64_eval_jnt.h5"
    test_dnndata_path = "./data/pair/SF1-TF1/jnt/blstm3_diff_64_test_jnt.h5"

    train_dnndata = HDF5(train_dnndata_path, mode='a')
    train_dnndata.save(jnt_diff_mcep[:train_stop_,:], ext='mcep')
    train_dnndata.save(train_len, ext='mcep_len')
    train_dnndata.save(X_mean, ext='X_mean')
    train_dnndata.save(X_std, ext='X_std')
    train_dnndata.close()

    eval_dnndata = HDF5(eval_dnndata_path, mode='a')
    eval_dnndata.save(jnt_diff_mcep[train_stop_:eval_stop_,:], ext='mcep')
    eval_dnndata.save(eval_len, ext='mcep_len')
    eval_dnndata.close()

    test_dnndata = HDF5(test_dnndata_path, mode='a')
    test_dnndata.save(jnt_diff_mcep[eval_stop_:,:], ext='mcep')
    test_dnndata.save(test_len, ext='mcep_len')
    test_dnndata.close()
    
if(ap_model == 0 and uv_model == 1):
    source_mcc = preprocessing.scale(jnt_mcep[:,:mgc_order], axis=0, with_mean=True,with_std=True)
    source_lp = jnt_mcep[:,40:40+66]
    source_uv = jnt_mcep[:,106:106+2]
    target_lp = jnt_mcep[:,108:108+66] # 66
    target_uv = jnt_mcep[:,174:174+2]
    target_mcc = jnt_mcep[:,176:216]
    
    jnt_mcep = None
    
    save_A = np.c_[source_mcc, source_lp]
    save_A = np.c_[save_A, source_uv]
    save_A = np.c_[save_A, target_lp]
    save_A = np.c_[save_A, target_uv]
    
    save_A = np.c_[save_A, target_mcc]
    
    source_mcc, source_ap, source_lp = None, None, None
    source_uv, target_lp, target_uv = None, None, None
    
    path5_train = "./data/pair/SF1-TF1/jnt/172_4_jnt.h5"
    path5_eval = "./data/pair/SF1-TF1/jnt/172_4_eval_jnt.h5"
    path5_test = "./data/pair/SF1-TF1/jnt/172_4_test_jnt.h5"
    
    save_data(save_A, path5_train, path5_eval, path5_test,
                    0, 1, train_stop_, eval_stop_)
    save_A = None
elif(ap_model == 1 and uv_model == 1): #and
    source_mcc = preprocessing.scale(jnt_mcep[:,:mgc_order], axis=0, with_mean=True,with_std=True)
    source_ap = preprocessing.scale(jnt_mcep[:,mgc_order:80], axis=0, with_mean=True,with_std=True)
    source_lp = jnt_mcep[:,80:80+66]
    source_uv = jnt_mcep[:,146:146+2]
    target_lp = jnt_mcep[:,148:148+66] # 66
    target_uv = jnt_mcep[:,214:214+2]
    target_mcc = jnt_mcep[:,216:216+40]
    target_ap = jnt_mcep[:,256:256+40]
    #target_mcc_N = preprocessing.scale(target_mcc, axis=0, with_mean=True,with_std=True)
    jnt_mcep = None
    
    
    '''
    save_F = np.c_[source_mcc, source_ap]
    save_F = np.c_[save_F, target_ap]
    
    save_D = np.c_[source_ap, target_ap]
    save_E = np.c_[source_ap, source_lp]
    save_E = np.c_[save_E, target_lp]
    save_E = np.c_[save_E, target_ap]
    '''
    
    #save_A = np.c_[source_mcc, source_ap]
    save_A = np.c_[source_mcc, source_lp]
    save_A = np.c_[save_A, source_uv]
    save_A = np.c_[save_A, target_lp]
    save_A = np.c_[save_A, target_uv]
    
    save_A = np.c_[save_A, target_mcc]
    #save_H = np.c_[save_A, target_ap]
    #save_A = None
    
    source_mcc, source_ap, source_lp = None, None, None
    source_uv, target_lp, target_uv = None, None, None

    '''
    path6_train = "./data/pair/SF1-TF1/jnt/ult_apmcc_jnt.h5"
    path6_eval = "./data/pair/SF1-TF1/jnt/ult_apmcc_eval_jnt.h5"
    path6_test = "./data/pair/SF1-TF1/jnt/ult_apmcc_test_jnt.h5"
    
    save_data(save_H, path6_train, path6_eval, path6_test,
                    0, 1, train_stop_, eval_stop_)
    save_H = None
    '''
    '''
    path4_train = "./data/pair/SF1-TF1/jnt/mccap2ap_jnt.h5"
    path4_eval = "./data/pair/SF1-TF1/jnt/mccap2ap_eval_jnt.h5"
    path4_test = "./data/pair/SF1-TF1/jnt/mccap2ap_test_jnt.h5"
    
    save_data(save_F, path4_train, path4_eval, path4_test,
                    0, 1, train_stop_, eval_stop_)
    save_F = None
    '''
    '''
    path2_train = "./data/pair/SF1-TF1/jnt/ap2ap_jnt.h5"
    path2_eval = "./data/pair/SF1-TF1/jnt/ap2ap_eval_jnt.h5"
    path2_test = "./data/pair/SF1-TF1/jnt/ap2ap_test_jnt.h5"
    
    save_data(save_D, path2_train, path2_eval, path2_test,
                    0, 1, train_stop_, eval_stop_)
    save_D = None
    
    path3_train = "./data/pair/SF1-TF1/jnt/aplp_jnt.h5"
    path3_eval = "./data/pair/SF1-TF1/jnt/aplp_eval_jnt.h5"
    path3_test = "./data/pair/SF1-TF1/jnt/aplp_test_jnt.h5"
    
    save_data(save_E, path3_train, path3_eval, path3_test,
                    0, 1, train_stop_, eval_stop_)
    save_E = None
    '''
    '''
    zero_pad = np.zeros([target_mcc.shape[0], 40])
    save_B = np.c_[save_A, target_mcc]
    save_B = np.c_[save_B, zero_pad]
    save_C = np.c_[save_A, zero_pad]
    save_C = np.c_[save_C, target_ap]
    save_A, zero_pad = None, None
    case1 = np.r_[save_B, save_C]
    save_B, save_C = None, None
    
    a = train_stop_
    b = eval_stop_ - train_stop_
    c = 2*a + 2*b
    
    path1_train = "./data/pair/SF1-TF1/jnt/joint216_jnt.h5"
    path1_eval = "./data/pair/SF1-TF1/jnt/joint216_eval_jnt.h5"
    path1_test = "./data/pair/SF1-TF1/jnt/joint216_test_jnt.h5"
    
    case2 = np.r_[case1[:a,:], case1[test_stop_:test_stop_+a,:]]
    train_dnndata = HDF5(path1_train, mode='a')
    train_dnndata.save(case2, ext='mcep')
    train_dnndata.save(train_len*2, ext='mcep_len')
    train_dnndata.close()
    case2 = None
        
    case2 = np.r_[case1[train_stop_:eval_stop_,:], case1[test_stop_+a:test_stop_+a+b,:]]
    eval_dnndata = HDF5(path1_eval, mode='a')
    eval_dnndata.save(case2, ext='mcep')
    eval_dnndata.save(eval_len*2, ext='mcep_len')
    eval_dnndata.close()
    case2 = None
    
    case2 = np.r_[case1[eval_stop_:test_stop_,:], case1[test_stop_+a+b:,:]]
    test_dnndata = HDF5(path1_test, mode='a')
    test_dnndata.save(case2, ext='mcep')
    test_dnndata.save(test_len*2, ext='mcep_len')
    test_dnndata.close()
    case2 = None
    case1 = None
    '''
elif(ap_model == 1):
    source_mcc = preprocessing.scale(jnt_mcep[:,:mgc_order], axis=0, with_mean=True,with_std=True)
    source_ap = preprocessing.scale(jnt_mcep[:,mgc_order+67:147], axis=0, with_mean=True,with_std=True)
    source_lp = jnt_mcep[:,mgc_order+1:mgc_order+67]
    target_mcc = jnt_mcep[:,(-2*mgc_order):(-1*mgc_order)]
    target_ap = jnt_mcep[:,(-1*mgc_order):]
    target_lp = jnt_mcep[:,147:213] # 66
    target_mcc_N = preprocessing.scale(target_mcc, axis=0, with_mean=True,with_std=True)
    jnt_mcep = None
    # mcepap -> mcepap *
    case1 = np.c_[source_ap, target_ap]
    path1_train = "./data/pair/SF1-TF1/jnt/ap80_jnt.h5"
    path1_eval = "./data/pair/SF1-TF1/jnt/ap80_eval_jnt.h5"
    path1_test = "./data/pair/SF1-TF1/jnt/ap80_test_jnt.h5"
    save_data(case1, path1_train, path1_eval, path1_test,
                    X_mean, X_std,
                    train_stop_, eval_stop_)
    case1 = None
    # mcepap + source mcc -> mcepap
    
    # mcepap + source mcc + target mcc -> mcepap *
    case3 = np.c_[source_ap, source_mcc]
    case3 = np.c_[case3, target_mcc_N]
    case3 = np.c_[case3, target_ap]
    path3_train = "./data/pair/SF1-TF1/jnt/ap160_jnt.h5"
    path3_eval = "./data/pair/SF1-TF1/jnt/ap160_eval_jnt.h5"
    path3_test = "./data/pair/SF1-TF1/jnt/ap160_test_jnt.h5"
    save_data(case3, path3_train, path3_eval, path3_test,
                    X_mean, X_std,
                    train_stop_, eval_stop_)
    case3 = None
    # mcepap + source mcc + target mcc + 語言參數 -> mcepap
    
'''
# save twfs
twfh5 = h5py.File('./data/pair/SF1-TF1/jnt/twf_BLSTM.h5', 'w')
twfh5.create_dataset('twf', data=twfs)
twfh5.close()
'''
