# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np 
import torch as th
import torch.utils.data as data
import torch.optim as optim
import argparse as ap
import operator
from sklearn import preprocessing

from os import path

import dnnVC
import scipy.sparse
import matplotlib.pyplot as plt

from dtw import dtw
from fastdtw import fastdtw
from dtw_c import dtw_c

import pyworld
import pysptk
from scipy.io import wavfile
from sprocket.util import HDF5, static_delta, melcd
from src.misc import low_cut_filter, transform_jnt
from sprocket.model import GV
#----------------------------------
import argparse

from sklearn.externals import joblib

from sprocket.model import GV, F0statistics, GMMConvertor
from sprocket.speech import FeatureExtractor, Synthesizer
import re
import math
import matplotlib.pyplot as plt

'''
parser = ap.ArgumentParser()
parser.add_argument("--source", action="store_true")
parser.add_argument("--target", action="store_true")
args = parser.parse_args()
'''
with open(path.join("configs","Configs_2.json")) as configs_file:
    configs = json.load(configs_file)
    
num_layers = 2
use_cuda = configs["use_cuda"]
hidden_size = 512

input_size = 40
output_size = 40

mgc_order=configs["mgc_order"]
shiftms = 0.005

device = th.device("cuda" if use_cuda else "cpu")    

def train(model, trn_dl, device, hidden):
    model.train()
    if th.cuda.is_available():
        trn_dl = trn_dl.cuda()
    print(trn_dl.shape)
    for i in range(trn_dl.shape[0]):
        print(trn_dl[i,:,:].shape)
        m_ = model.forward(trn_dl[i,:,:], hidden, True)
        temp = m_.cpu()
        temp2 = temp.detach().numpy()
        if(i == 0):
            y_head = temp2
        else:
            y_head = np.r_[y_head, temp2]
    '''
    for i in range(trn_dl.shape[0]):
        m_ = model.forward(trn_dl[i,:])
        
        temp = m_.cpu()
        if (i == 0):
            mean = temp.detach().numpy()
        else:
            temp_2 = temp.detach().numpy()
            mean = np.c_[mean, temp_2]
    '''
    return y_head

def zero_padding(jnt, length_max):
    a = jnt.shape
    #print(a)
    output_2 = np.zeros([1,length_max,a[1]])
    zero_pad = np.zeros([length_max - a[0], a[1]])
    output = np.r_[jnt, zero_pad]
    output_2[0,:,:] = output
    return output_2

def lstm_data(mcep_, length_max):
    a = mcep_.shape
    print(a)
    temp_num = int(mcep_.shape[0] / length_max)
    temp_num2 = mcep_.shape[0] % length_max
    output = mcep_
    #print(temp_num)
    if(temp_num2 != 0):
        temp_2 = length_max*(temp_num+1) - a[0]
        zero_pad = np.zeros([temp_2, a[1]])
        output = np.r_[output, zero_pad]
    output_2 = np.reshape(output, (temp_num+1, length_max, a[1]))
    #print(output_2.shape)
    return output_2

def normalized(jnt, X_mean, X_std):
    jnt_out = jnt[:,:mgc_order]
    jnt_out = (jnt_out - X_mean) / X_std
    jnt_out = np.c_[jnt_out, jnt[:,mgc_order:]]
    return jnt_out

def read_labs(listf):
    starts, stops, labs = [], [], []
    f_ = listf.rstrip()
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
            
    print(len(starts[0]))
    print(len(stops[0]))
    print(len(labs[0]))
    return starts, stops, labs

def data_for64model(mceps, starts, stops, labs, phone_table):
    for i in range(len(starts)):
        K = 0
        out_mcep = None
        mcep_t = mceps
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
                if (aa != bb):
                    print(starts[i][k], stops[i][k], labs[i][k])
                    print(mcep_t[start_frame:stop_frame+1,:].shape)
                    print(zero_pad.shape)
                    print(start_frame, stop_frame, frame_len)
                    zero_pad = zero_pad[:aa,:]
                mcep_milk = np.c_[mcep_t[start_frame:stop_frame+1,:], zero_pad]
                #mcep_milk = np.c_[mcep_milk, zero_pad]
                if(K==0):
                    out_mcep = mcep_milk
                else:
                    out_mcep = np.r_[out_mcep, mcep_milk]
                K = K + 1
        outputmceps = out_mcep
    return outputmceps

def estimate_twf(orgdata, tardata, distance='melcd', fast=True, otflag=None):
    """time warping function estimator

    Parameters
    ---------
    orgdata : array, shape(`T_org`, `dim`)
        Array of source feature
    tardata : array, shape(`T_tar`, `dim`)
        Array of target feature
    distance : str, optional
        distance function
        `melcd` : mel-cepstrum distortion
    fast : bool, optional
        Use fastdtw instead of dtw
        Default set to `True`
    otflag : str,
        Perform alignment into either original or target length
        `org` : align into original length
        `tar` : align into target length
        Default set to None

    Returns
    ---------
    twf : array, shape(`2`, `T`)
        Time warping function between original and target
    """

    if distance == 'melcd':
        def distance_func(x, y): return melcd(x, y)
    else:
        raise ValueError('other distance metrics than melcd does not support.')

    if otflag is None:
        # use dtw or fastdtw
        if fast:
            _, path = fastdtw(orgdata, tardata, dist=distance_func)
            twf = np.array(path).T
        else:
            _, _, _, twf = dtw(orgdata, tardata, distance_func)
    else:
        # use dtw_c to align target/original feature vector
        ldim = orgdata.shape[1] - 1
        if otflag == 'org':
            _, twf, _, _ = dtw_c.dtw_org_to_trg(tardata, orgdata,
                                                0, ldim, 5.0, 100.0, 100.0)
        else:
            _, twf, _, _ = dtw_c.dtw_org_to_trg(orgdata, tardata,
                                                0, ldim, 5.0, 100.0, 100.0)
        #print("H")
        #print(twf.shape)
        #print("H")
        twf[:, 1] = np.array(range(twf.shape[0]))  # replace target index by frame number
        twf = twf.T
        #print(twf.shape)
        if otflag == 'org':
            twf = twf[::-1, :]  # swap cols
            #print(twf.shape)
            #assert twf.shape[0] == orgdata.shape[0]
        else:
            assert twf.shape[1] == tardata.shape[0]

    return twf

def mcd_funct(target, source):
    if(target.shape[0] == source.shape[0]):
        diff = target - source
    else:
        twf = estimate_twf(source, target,
                           distance='melcd', otflag=None)
        sor_mcep = source[twf[0]]
        tar_mcep = target[twf[1]]
        diff = tar_mcep - sor_mcep
    
    mcd_ = 10.0 / 2.302585 * np.sqrt(2.0 * np.sum(diff * diff, 1))
    output = np.mean(mcd_)
    
    return output

def pesq_process(pesq_path, sor_mcep):
    f = pesq_path.rstrip()
    wavf = os.path.join(f + '.wav')
    fs, x = wavfile.read(wavf)
    x = x.astype(np.float)
    x = low_cut_filter(x, fs, cutoff=70)
    assert fs == sampling_rate
    
    f0, spc, ap = feat.analyze(x)
    mcep = feat.mcep(dim=configs["mgc_order"], alpha=configs["alpha_value"])
    
    # dtw
    #print(sor_mcep.shape)
    #print(mcep[:,1:].shape)
    
    A = np.array(sor_mcep)
    B = np.array(mcep[:,1:])
    #print(A.shape)
    #print(B.shape)

    twf = estimate_twf(A, B,
                           distance='melcd', otflag='org')
    '''
    twf = estimate_twf(sor_mcep, mcep[:,1:],
                           distance='melcd', otflag=None)
    '''
    out_mcep = mcep[twf[1]]
    return out_mcep[:,1:], mcep[:,1:]

def plot_ap(f, ap, cvap, index, mcepap, cvmcepap):
    Om = range(ap.shape[1])
    #Om = np.linspace(0, np.pi, num=ap.shape[1])
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    print(ap.shape)
    print(cvap.shape)
    plt.plot(Om, sum_ap(ap), label=r'spc')
    plt.plot(Om, sum_ap(cvap), label=r'cv spc')
    plt.title('convert spc')
    plt.legend()
    plt.grid()
    plt.subplot(122)
    print(mcepap.shape)
    print(cvmcepap.shape)
    xi = range(mcepap.shape[1])
    plt.plot(xi, mcepap[index], label=r'mcep')
    plt.plot(xi, cvmcepap[index], label=r'cv mcep')
    plt.title('convert mcep')
    plt.xlabel(r'dim')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    figpath = os.path.join( f + '.png')
    plt.savefig(figpath)
    plt.close()

def sum_ap(ap):
    out = np.zeros(ap.shape[1])
    for i in range(ap.shape[1]):
        out[i] = sum(ap[:,i])
    return out

def add_uv(mceps, f0s):
    temp_B = math.log(550)
    K = 0
    mcep_A = mceps
    f0_A = f0s
    #print(mcep_A.shape)
    #print(f0_A.shape)
    #print(len(mceps))
    for i in range(1):
        
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
        #sys.exit("stop")
        zero_pad[:, 1] = zero_pad[:, 1] / temp_B

        out_A = np.c_[mcep_A, zero_pad]
        
        K = K + 1
    return out_A

argv = sys.argv

shortest = 0
pesq = 0
output_mcd = 0
ap_model = 0
temp_len = 0
apuv = 0
for i in range(1, len(argv)):
    if (operator.eq(argv[i], "shortest")):
        shortest = 1
    elif(operator.eq(argv[i], "pesq")):
        pesq = 1
    elif(operator.eq(argv[i], "mcd")):
        output_mcd = 1
    elif(operator.eq(argv[i], "ap")):
        ap_model = 1
    elif(operator.eq(argv[i], "aplp")):
        ap_model = 2
    elif(operator.eq(argv[i], "apmcc")):
        ap_model = 3
    elif(operator.eq(argv[i], "apuv")):
        apuv = 1
    if(temp_len<len(argv[i])):
        temp_len = len(argv[i])
        temp_index = i
        
mcd_path = argv[temp_index]
print(mcd_path)

if (operator.eq(argv[1], "None") or operator.eq(argv[1], "diff2")):
    tarstatspath = configs["var_path"]
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC.model"
elif (operator.eq(argv[1], "diff")):
    tarstatspath = configs["var_diff_path"] # for diff
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_diff.model"
elif (operator.eq(argv[1], "diffdnn3")):
    tarstatspath = configs["dnn3_var_diff_path"] # for diff dnn3
    state_dict_path = configs["state_dict_path2"] # for diff dnn3

if (operator.eq(argv[3], "200mcd")):
    max_epoch = configs["max_epoch"]
    length_max = 512
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_200mcd.model"
elif(operator.eq(argv[3], "50mse")):
    max_epoch = 50
    length_max = 512
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_50mse.model"
elif(operator.eq(argv[3], "200mseZ")):
    max_epoch = configs["max_epoch"]
    length_max = 10740
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_200mseZ.model"
elif(operator.eq(argv[3], "200mse")):
    max_epoch = configs["max_epoch"]
    length_max = 512
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC.model"
elif(operator.eq(argv[3], "64model")):
    input_size = 172
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_64.model"
elif(operator.eq(argv[3], "64fast")):
    input_size = 172
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_64fast.model"
elif(operator.eq(argv[3], "64slow")):
    input_size = 172
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_64slow.model"
elif(operator.eq(argv[3], "64threedata")):
    input_size = 172
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_64three.model"
elif(operator.eq(argv[3], "ult")):
    input_size = 216
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_ult_mcc.model"
    ap_model = 4
elif(operator.eq(argv[3], "onlyuv")):
    input_size = 176
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_onlyuv.model"
elif(operator.eq(argv[3], "apuv")):
    print("Here is apuv")
    input_size = 216
    output_size = 80
    num_layers = 3
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_apuv.model"
elif(operator.eq(argv[3], "allin")):
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_allin.model"
elif(operator.eq(argv[3], "others")):
    pass

# special case #############################
if(operator.eq(argv[1], "diff") and operator.eq(argv[3], "64model")):
    input_size = 172
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_64diff.model"
# read target variance #########################
tarstats_h5 = HDF5(tarstatspath, mode='r')
targvstats_var = tarstats_h5.read(ext='gv')
if (operator.eq(argv[3], "linear")):
    targvstats_mean = tarstats_h5.read(ext='mean') # (80,)
    targvpost_mean = targvstats_mean[:mgc_order]
tarstats_h5.close()
#targvstats_var = 0.1 * targvstats_var
#variance_ = th.from_numpy(targvstats_var)
#variance = variance_.float()
# read target variance for postfilter #################
gv_post_path = configs["gv_path"]
gvpost_h5 = HDF5(gv_post_path, mode='r')
targvpost_var = gvpost_h5.read(ext='gv')
targvpost_std = targvpost_var[1,1:]
gvpost_h5.close()
'''
test2 = np.eye(targvstats_var.shape[0], dtype=float)
for mm in range(targvstats_var.shape[0]):
    test2[mm, mm] = targvstats_var[mm]
cov_inv = np.linalg.inv(test2)
'''
# read cvgv ###############################
cvgvstatspath = configs["cvgv_path"]
cvgvstats_h5 = HDF5(cvgvstatspath, mode='r')
cvgvstats = cvgvstats_h5.read(ext='cvgv')
diffcvgvstats = cvgvstats_h5.read(ext='diffcvgv')
cvgvstats_h5.close()

# read phoneme table #########################
phone_table = []
phone_table_file = open("./data/phoneme_table.txt","r")
sents_A = phone_table_file.readlines()
for i in range(len(sents_A)):
    temp_sents_A = re.split("\\n", sents_A[i])
    phone_table.append(temp_sents_A[0])
phone_table_file.close()
print(phone_table)

# linear transform ###################################
if (operator.eq(argv[3], "linear")):
    x_file = HDF5("./data/pair/SF1-TF1/jnt/blstm3_jnt.h5", mode='r')
    sor_mean = x_file.read(ext='X_mean')
    sor_std = x_file.read(ext='X_std')
    x_file.close()

# read model
model = dnnVC.BiRNN( input_size, hidden_size, output_size, num_layers).to(device)
source_state_dict = th.load(state_dict_path)
model.load_state_dict(source_state_dict)
if (ap_model == 1): # read ap model
    ap_model_path = "./data/pair/SF1-TF1/model/BLSTM_VC_ap2ap.model"
    APmodel = dnnVC.BiRNN( mgc_order, hidden_size, output_size, num_layers).to(device)
    ap_model_dict = th.load(ap_model_path)
    APmodel.load_state_dict(ap_model_dict)
elif (ap_model == 2): # read ap model
    ap_model_path = "./data/pair/SF1-TF1/model/BLSTM_VC_aplp.model"
    APmodel = dnnVC.BiRNN( 172, hidden_size, output_size, num_layers).to(device)
    ap_model_dict = th.load(ap_model_path)
    APmodel.load_state_dict(ap_model_dict)
elif (ap_model == 3): # read ap model
    ap_model_path = "./data/pair/SF1-TF1/model/BLSTM_VC_apmcc.model"
    APmodel = dnnVC.BiRNN( 80, hidden_size, output_size, num_layers).to(device)
    ap_model_dict = th.load(ap_model_path)
    APmodel.load_state_dict(ap_model_dict)
elif (ap_model == 4): # read ap model
    ap_model_path = "./data/pair/SF1-TF1/model/BLSTM_VC_ult_apmcc.model"
    APmodel = dnnVC.BiRNN(input_size, hidden_size, output_size, num_layers).to(device)
    ap_model_dict = th.load(ap_model_path)
    APmodel.load_state_dict(ap_model_dict)

mcepgv = GV()

sampling_rate = configs["sampling_rate"]
# constract FeatureExtractor class
feat = FeatureExtractor(analyzer=configs["analyzer"],
                            fs=sampling_rate,
                            fftl=configs["wav_fftl"],
                            shiftms=configs["wav_shiftms"],
                            minf0=configs["f0_minf0"],
                            maxf0=configs["f0_maxf0"])

# constract Synthesizer class
synthesizer = Synthesizer(fs=sampling_rate,
                              fftl=configs["wav_fftl"],
                              shiftms=configs["wav_shiftms"])

# mcd
#mcd_file = open(args.mcd_dir,'w')

# zero ap
#ap_0 = np.zeros((ap.shape[0],ap.shape[1]), dtype=np.float64)

#h5_dir = configs["_path"]
# conversion in each evaluation file
mean_data = []
if (operator.eq(argv[2], "eval")):
    converted_list = configs["eval_list_file"]
elif (operator.eq(argv[2], "test")):
    converted_list = configs["test_list_file"]
    pesq_list = configs["target_test_list"]
elif (operator.eq(argv[2], "total")):
    converted_list = configs["total_list_file"]
elif (operator.eq(argv[2], "CEspell") or operator.eq(argv[2], "CEword")):
    if (operator.eq(argv[2], "CEspell")):
        converted_list = "/home/icee23/paper/20190312/CEMIX_spell/CEMIX_spell.lst"
        mean_std_path = "/home/icee23/paper/20190312/CEMIX_spell/CEMIX_spell_Ndata.h5"
    elif (operator.eq(argv[2], "CEword")):
        converted_list = "/home/icee23/paper/20190312/data/CEMIX_word_16k.lst"
        mean_std_path = "/home/icee23/paper/20190312/data/CEMIX_word_Ndata.h5"
    mean_std_file = HDF5(mean_std_path, mode='r')
    X_mean = mean_std_file.read(ext='X_mean')
    X_std = mean_std_file.read(ext='X_std')
    mean_std_file.close()

if(pesq == 1):
    pesq_file = open(pesq_list,"r")
    pesq_path = pesq_file.readlines()
with open(converted_list, 'r') as fp:
    K = 0
    for line in fp:
        # open wav file
        f = line.rstrip()
        #h5f = os.path.join(h5_dir, f + '_cv.h5') ##
        wavf = os.path.join(f + '.wav')
        fs, x = wavfile.read(wavf)
        x = x.astype(np.float)
        x = low_cut_filter(x, fs, cutoff=70)
        assert fs == sampling_rate
        
        # analyze F0, mcep, and ap
        f0, spc, ap = feat.analyze(x)
        mcep = feat.mcep(dim=configs["mgc_order"], alpha=configs["alpha_value"])
        mcep_0th = mcep[:, 0]
        #codeap = feat.codeap()
        
        # pesq data
        if(pesq == 1):
            pesq_mcep_, target_mcc = pesq_process(pesq_path[K], mcep[:, 1:])
            pesq_mcep = np.c_[mcep_0th, pesq_mcep_]
            
        # f0 rapt
        _f0_rapt = pysptk.rapt(x.astype(np.float32), fs,
                                      hopsize=(configs["wav_shiftms"]*fs/1000),
                                      min=configs["f0_minf0"], max=configs["f0_maxf0"], otype="f0")
        f0_rapt1 = _f0_rapt.astype(np.float64)
        
        #print('f0.shape : ')
        #print(f0.shape[0])
        #print('f0_rapt.shape : ')
        #print(f0_rapt1.shape[0])
        
        f0_rapt = np.zeros(f0.shape[0])
        if (f0_rapt1.shape[0] > f0.shape[0]):
            i = 0
            while (i<f0.shape[0]):
                f0_rapt[i] = f0_rapt1[i]
                i += 1
        elif (f0_rapt1.shape[0] < f0.shape[0]):
            i = 0
            while (i<f0.shape[0]):
                if (i >= f0_rapt1.shape[0]):
                    f0_rapt[i] = 0.0
                else:
                    f0_rapt[i] = f0_rapt1[i]
                i += 1
        else: #f0_rapt1.shape[0] == f0.shape[0]
            f0_rapt = f0_rapt1

        # convert F0
        #cvf0 = f0stats.convert(f0, orgf0stats, tarf0stats)
        
        if K>=5 and shortest == 1:
            print("short test!")
            break
        
        # converted mcep
        mcep_ = mcep[:, 1:]
        print(mcep[:, 1:].shape)
        if (operator.eq(argv[2], "CEspell") or operator.eq(argv[2], "CEword")):
            mcep_N = normalized(mcep_, X_mean[1:], X_std[1:])
        elif (operator.eq(argv[3], "64model") or operator.eq(argv[3], "64fast") or operator.eq(argv[3], "64slow") or operator.eq(argv[3], "64threedata")):
            lab_path = "/home/icee23/paper/20180919/sprocket-master/example/data/XLAB_kiwi/" + f[72:]
            starts, stops, labs = read_labs(lab_path)
            mcep_N_ = preprocessing.scale(mcep_, axis=0, with_mean=True,with_std=True)
            mcep_N = data_for64model(mcep_N_, starts, stops,
                                                      labs, phone_table)
            mcep_N = np.c_[mcep_N, mcep_N[:,-66:]]
        elif(apuv == 1 or operator.eq(argv[3], "ult")):
            #mcep
            mcep_N_ = preprocessing.scale(mcep_, axis=0, with_mean=True,with_std=True)
            #mcepap
            mcepap = pysptk.sp2mc(ap, mgc_order, configs["alpha_value"])
            ap_0th = mcepap[:, 0]
            ap_ = mcepap[:, 1:]
            print("mcepap.shape")
            print(mcepap.shape)
            ap_N = preprocessing.scale(ap_, axis=0, with_mean=True,with_std=True)
            #66+2
            lab_path = "/home/icee23/paper/20180919/sprocket-master/example/data/XLAB_kiwi/" + f[72:]
            starts, stops, labs = read_labs(lab_path)
            mcep_N = np.c_[mcep_N_, ap_N]
            print("mcep+mcepap = 80")
            print(mcep_N.shape)
            mcep_N = data_for64model(mcep_N, starts, stops,
                                                      labs, phone_table)
            print("mcep+mcepap+66 = 146")
            print(mcep_N.shape)
            mcep_N = add_uv(mcep_N, f0_rapt)
            print("mcep+mcepap+66+uv = 148")
            print(mcep_N.shape)
            mcep_N = np.c_[mcep_N, mcep_N[:,-68:]]
            print("mcep+mcepap+66+uv+66+uv = 216")
        elif(operator.eq(argv[3], "onlyuv")):
            mcep_N_ = preprocessing.scale(mcep_, axis=0, with_mean=True,with_std=True)
            lab_path = "/home/icee23/paper/20180919/sprocket-master/example/data/XLAB_kiwi/" + f[72:]
            starts, stops, labs = read_labs(lab_path)
            print(mcep_N_.shape)
            mcep_N = data_for64model(mcep_N_, starts, stops,
                                                      labs, phone_table)
            print(mcep_N.shape)
            mcep_N = add_uv(mcep_N, f0_rapt)
            print(mcep_N.shape)
            mcep_N = np.c_[mcep_N, mcep_N[:,-68:]]
            print(mcep_N.shape)
        else:
            mcep_N = preprocessing.scale(mcep_, axis=0, with_mean=True,with_std=True)
        mcep_re = np.reshape(mcep_N, (1, mcep_N.shape[0], mcep_N.shape[1]))
        #if (operator.eq(argv[3], "200mse")):
            # mcep_re = lstm_data(mcep_N, length_max)
            #mcep_re = np.reshape(mcep_N, (1, mcep_N.shape[0], mcep_N.shape[1]))
        #elif (operator.eq(argv[3], "200mseZ")):
            #mcep_re = zero_padding(mcep_N, length_max)
        mcep_in_D = th.from_numpy(mcep_re)
        mcep_data = mcep_in_D.float()
        print(mcep_N.shape)
        #print(mcep_re.shape)
        if (operator.eq(argv[3], "linear")):
            cvmcep_wopow = (mcep_ - sor_mean)/sor_std*targvpost_std + targvpost_mean
        else:
            cvmcep_wopow = train(model, mcep_data, device, None)
        print(cvmcep_wopow.shape)
        print(mcep_0th.shape)
        #print(mcep_[1,:])
        #print(cvmcep_wopow[1,:])
        if(apuv == 1):
            cvmcep = np.c_[mcep_0th, cvmcep_wopow[:,:mgc_order]]
            cvmcepap = np.c_[ap_0th, cvmcep_wopow[:,mgc_order:]]
            print("cvmcepap.shape")
            print(cvmcepap.shape)
            cvap = pysptk.mc2sp(cvmcepap, configs["alpha_value"], configs["wav_fftl"])
            plot_ap(f, ap, cvap, 3000, mcepap, cvmcepap)
            cvspc = pysptk.mc2sp(cvmcep, configs["alpha_value"], configs["wav_fftl"])
            f = os.path.join( f + '_mcep')
            plot_ap(f, spc, cvspc, 3000, mcep, cvmcep)
            #ap = pysptk.mc2sp(mcepap, configs["alpha_value"], configs["wav_fftl"])
            ap = cvap
            print("ap = cvap")
            #cvmcep = mcep
        else:
            cvmcep = np.c_[mcep_0th, cvmcep_wopow]
        
        # convert ap
        if(ap_model != 0):
            mcepap = pysptk.sp2mc(ap, mgc_order, configs["alpha_value"])
            ap_0th = mcepap[:, 0]
            ap_ = mcepap[:, 1:]
            print(mcepap[:, 1:].shape)
            if(ap_model == 1):
                ap_N = preprocessing.scale(ap_, axis=0, with_mean=True,with_std=True)
            elif(ap_model == 2):
                lab_path = "/home/icee23/paper/20180919/sprocket-master/example/data/XLAB_kiwi/" + f[72:]
                starts, stops, labs = read_labs(lab_path)
                ap_N_ = preprocessing.scale(ap_, axis=0, with_mean=True,with_std=True)
                ap_N = data_for64model(ap_N_, starts, stops,
                                                      labs, phone_table)
                ap_N = np.c_[ap_N, ap_N[:,-66:]]
            elif(ap_model == 3):
                f = os.path.join( f + '_apmcc')
                ap_N_ = np.c_[mcep_, ap_]
                ap_N = preprocessing.scale(ap_N_, axis=0, with_mean=True,with_std=True)
            ap_re = np.reshape(ap_N, (1, ap_N.shape[0], ap_N.shape[1]))
            print(ap_re.shape)
            ap_in_D = th.from_numpy(ap_re)
            ap_data = ap_in_D.float()
            if(ap_model == 4):
                ap_data = mcep_data
            cvap_wopow = train(APmodel, ap_data, device, None)
            cvmcepap = np.c_[ap_0th, cvap_wopow]
            
            cvap = pysptk.mc2sp(cvmcepap, configs["alpha_value"], configs["wav_fftl"])
            
            #plot_ap(f, ap, cvap, 3000, mcepap, cvmcepap)
            #ap = cvap
            print("ap = cvap")
        
        # mcd
        if(output_mcd == 1 and pesq == 1):
            mcd_a = mcd_funct(target_mcc, mcep_)
            mcd_b = mcd_funct(cvmcep_wopow, mcep_)
            mcd_c = mcd_funct(target_mcc, cvmcep_wopow)
            mcd_temp = np.zeros([3,1])
            mcd_temp[0] = mcd_a
            mcd_temp[1] = mcd_b
            mcd_temp[2] = mcd_c
            if(K == 0):
                #output_mceps = cvmcep_wopow
                output_mcds = mcd_temp
            else:
                #output_mceps = np.r_[output_mceps, cvmcep_wopow]
                output_mcds = np.c_[output_mcds, mcd_temp]
        K = K + 1
        #print(mcep[4744,:])
        #print(cvmcep[4744,:])
        
        #break

        # save converted features into a hdf5 file
        '''
        h5 = HDF5(h5f, mode='a')
        h5.save(f0_rapt, ext='f0')
        h5.save(cvf0, ext='cvf0')
        h5.save(spc, ext='spc')
        h5.save(cvspc, ext='cvspc')
        h5.save(ap, ext='ap')
        h5.save(cvap, ext='cvap')
        h5.close()
        '''
        # synthesis VC w/ GV
        gv_name = ''
        ap_name = ''
        if (operator.eq(argv[1], "None")):
            if(operator.eq(argv[4], "gv")):
                cvmcep_wGV = mcepgv.postfilter(cvmcep,
                                               targvpost_var,
                                               cvgvstats=cvgvstats,
                                               alpha=0,
                                               startdim=1)
            elif(operator.eq(argv[4], "nogv")):
                cvmcep_wGV = cvmcep
            
            wav = synthesizer.synthesis(f0_rapt,
                                            cvmcep_wGV,
                                            ap,
                                            rmcep=mcep,# rmcep=mcep
                                            alpha=configs["alpha_value"],
                                            )
            if (operator.eq(argv[2], "CEspell")):
                wavpath = os.path.join( f + '_BLSTM_CEspell_VC.wav')
            elif (operator.eq(argv[2], "CEword")):
                wavpath = os.path.join( f + '_BLSTM_CEword_VC.wav')
            elif (operator.eq(argv[3], "onlyuv")):
                if(operator.eq(argv[4], "gv")):
                    gv_name = 'gv_'
                wavpath = os.path.join( f + '_BLSTM_onlyuv_' + ap_name + gv_name + 'VC.wav')
            elif (operator.eq(argv[3], "64model") or operator.eq(argv[3], "ult")):
                if(operator.eq(argv[4], "gv")):
                    gv_name = 'gv_'
                if(ap_model == 1):
                    ap_name = 'ap_'
                elif(ap_model == 2):
                    ap_name = 'aplp_'
                elif(ap_model == 3):
                    ap_name = 'apmcc_'
                    
                if(operator.eq(argv[3], "ult")):
                    wavpath = os.path.join( f + '_BLSTM_ult_' + ap_name + gv_name + 'VC.wav')
                else:
                    wavpath = os.path.join( f + '_BLSTM_64model_' + ap_name + gv_name + 'VC.wav')
                '''
                print("spc")
                print(spc.shape)
                print("cvspc")
                print(cvmcep_wGV.shape)
                '''
                #cvjoswe = pysptk.mc2sp(cvmcep_wGV, configs["alpha_value"], configs["wav_fftl"])
                #f1 = os.path.join( f + '_64model_mcep')
                #plot_ap(f1, spc, cvjoswe, 3000, mcep, cvmcep)
            elif (operator.eq(argv[3], "64fast")):
                if(operator.eq(argv[4], "gv")):
                    wavpath = os.path.join( f + '_BLSTM_64fast_gv_VC.wav')
                else:
                    wavpath = os.path.join( f + '_BLSTM_64fast_VC.wav')
            elif (operator.eq(argv[3], "64slow")):
                if(operator.eq(argv[4], "gv")):
                    wavpath = os.path.join( f + '_BLSTM_64slow_gv_VC.wav')
                else:
                    wavpath = os.path.join( f + '_BLSTM_64slow_VC.wav')
            elif (operator.eq(argv[3], "64threedata")):
                if(operator.eq(argv[4], "gv")):
                    wavpath = os.path.join( f + '_BLSTM_64three_gv_VC.wav')
                else:
                    wavpath = os.path.join( f + '_BLSTM_64three_VC.wav')
            elif (operator.eq(argv[3], "allin")):
                wavpath = os.path.join( f + '_BLSTM_allin_VC.wav')
            elif (operator.eq(argv[3], "linear")):
                wavpath = os.path.join( f + '_linear_VC.wav')
            elif (operator.eq(argv[3], "apuv")):
                wavpath = os.path.join( f + '_apuv.wav')
            elif (ap_model == 1):
                wavpath = os.path.join( f + '_ap2ap.wav')
            elif (ap_model == 2):
                wavpath = os.path.join( f + '_aplp.wav')
            elif(ap_model == 3):
                wavpath = os.path.join( f + '_apmcc.wav')
            else:
                wavpath = os.path.join( f + '_BLSTM_dtwblstm_VC.wav')
            '''
            #print mcc
            x1 = np.arange(41)
            k_range = round(mcep.shape[0]/2)
            KK = np.arange(k_range, k_range+10)
            print(k_range)
            print(k_range+10)
            for k in KK:
                picpath = os.path.join( f + '_dnnVC_' + repr(k) + '.png')
                #plt.plot(x1,cvmcep[k,:],"--")
                #plt.plot(x1,mcep[k,:],"r--")
                plt.scatter(x1, mcep[k,:], c='red', marker='o')
                plt.scatter(x1, cvmcep[k,:], c='blue', marker='o')
            
                plt.ylabel("coef")
                plt.xlabel("mcc dim")
                plt.title("Converted mcc")
            
                plt.savefig(picpath)
                picpath = None
                plt = None
            '''
            
            # synthesis DIFFVC w/ GV
        if (operator.eq(argv[1], "diff") or operator.eq(argv[1], "diffdnn3") or operator.eq(argv[1], "diff2")):

            if(operator.eq(argv[4], "gv")):
                
                if(operator.eq(argv[1], "diff")):
                    cvmcep[:, 0] = 0.0
                    cvmcep_wGV = mcepgv.postfilter(mcep + cvmcep,
                                               targvpost_var,
                                               cvgvstats=None,
                                               alpha=1,
                                               startdim=1)
                    xi = range(mcep.shape[0])
                    diff = cvmcep_wGV - mcep
                    distance = np.sqrt(np.sum(diff * diff, 1) / 41)
                    #mcd_ = 10.0 / 2.302585 * np.sqrt(2.0 * np.sum(diff * diff, 1))
                    plt.plot(xi, distance, label=r'mcd')
                    #plt.plot(xi, cvmcep_wGV[3000], label=r'diff1 mcep')
                    plt.title('diff1 mcd by frame')
                    plt.xlabel(r'frame')
                    plt.legend()
                    plt.grid()
                    plt.tight_layout()
                    figpath = os.path.join( f + '_diff.png')
                    plt.savefig(figpath)
                    plt.close()
                elif(operator.eq(argv[1], "diff2")):
                    cvmcep = cvmcep - mcep
                    cvmcep[:, 0] = 0.0
                    cvmcep_wGV = mcepgv.postfilter(mcep + cvmcep,
                                               targvpost_var,
                                               cvgvstats=None,
                                               alpha=1,
                                               startdim=1)# - mcep
                    xi = range(mcep.shape[1])
                    plt.plot(xi, mcep[3000], label=r'mcep')
                    plt.plot(xi, cvmcep_wGV[3000], label=r'diff2 mcep')
                    plt.title('diff2 mcep')
                    plt.xlabel(r'dim')
                    plt.legend()
                    plt.grid()
                    plt.tight_layout()
                    figpath = os.path.join( f + '_diff2.png')
                    plt.savefig(figpath)
                    plt.close()
            elif(operator.eq(argv[4], "nogv")):
                if(operator.eq(argv[1], "diff")):
                    cvmcep[:, 0] = 0.0
                    cvmcep_wGV = mcep + cvmcep
                else:
                    cvmcep_wGV = cvmcep
                
            wav = synthesizer.synthesis_diff(x,
                                                 cvmcep_wGV,
                                                 rmcep=mcep,
                                                 alpha=configs["alpha_value"],
                                                 )
            if (operator.eq(argv[2], "CEspell")):
                wavpath = os.path.join( f + '_diffBLSTM_CEspell_VC.wav')
            elif (operator.eq(argv[2], "CEword")):
                wavpath = os.path.join( f + '_diffBLSTM_CEword_VC.wav')
            elif (operator.eq(argv[3], "64model")):
                if(operator.eq(argv[4], "gv")):
                    wavpath = os.path.join( f + '_BLSTM_64diff_gv_VC.wav')
                else:
                    wavpath = os.path.join( f + '_BLSTM_64diff_VC.wav')
            elif(operator.eq(argv[1], "diff")):
                if(operator.eq(argv[4], "gv")):
                    wavpath = os.path.join( f + '_BLSTM_DIFF1_gv_VC.wav')
            elif(operator.eq(argv[1], "diff2")):
                if(operator.eq(argv[4], "gv")):
                    wavpath = os.path.join( f + '_BLSTM_DIFF2_gv_VC.wav')
            else:
                wavpath = os.path.join( f + '_diff_dtwblstmVC_0519.wav')
        
        if (pesq == 1 and pesq == 0):
            pesq_wav = synthesizer.synthesis(f0_rapt,
                                            pesq_mcep,
                                            ap,
                                            rmcep=None,# rmcep=mcep
                                            alpha=configs["alpha_value"],
                                            )
            pesq_output_path = os.path.join( f + '_pesq.wav')
            pesq_wav = np.clip(pesq_wav, -32768, 32767)
            wavfile.write(pesq_output_path, fs, pesq_wav.astype(np.int16))
            print(pesq_output_path)
        
        # write waveform
        wav = np.clip(wav, -32768, 32767)
        wavfile.write(wavpath, fs, wav.astype(np.int16))
        print(wavpath)
if(pesq == 1):
    pesq_file.close()

better = 0
worse = 0
equal = 0
worst = 0
if(output_mcd == 1):
    output_mcds = output_mcds.T
    fp = open(mcd_path, "w")
    print(output_mcds.shape)
    for i in range(output_mcds.shape[0]):
        for j in range(output_mcds.shape[1]):
            fp.write(str(output_mcds[i, j]))
            fp.write(" ")
        fp.write("\n")
        
        if(output_mcds[i, 1] > output_mcds[i, 0] or output_mcds[i, 2] > output_mcds[i, 0]):
            worst = worst + 1
        else:
            if(output_mcds[i, 1] == output_mcds[i, 2]):
                equal = equal + 1
            elif(output_mcds[i, 0] > output_mcds[i, 1] and output_mcds[i, 1] > output_mcds[i, 2]):
                better = better + 1
            else:
                worse = worse + 1
        
    fp.write("\n")
    fp.write(str(better))
    fp.write(" ")
    fp.write(str(equal))
    fp.write(" ")
    fp.write(str(worse))
    fp.write(" ")
    fp.write(str(worst))
    fp.write("\n")
    fp.write("file number : ")
    fp.write(str(output_mcds.shape[0]))
    fp.close()


