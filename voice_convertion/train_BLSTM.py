# -*- coding: utf-8 -*-

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

from os import path

import dnnVC

import pyworld
import pysptk
from scipy.io import wavfile
from sprocket.util import HDF5, static_delta
from src.misc import transform_jnt
'''
parser = ap.ArgumentParser()
parser.add_argument("--source", action="store_true")
parser.add_argument("--target", action="store_true")
args = parser.parse_args()
'''
with open(path.join("configs","Configs_2.json")) as configs_file:
    configs = json.load(configs_file)
    
max_epoch = 20
batch_size = 1
num_layers = 2 #3
use_cuda = configs["use_cuda"]
hidden_size = 512

input_size = 40
output_size = 40

gmmmode = configs["gmmmode"]

argv = sys.argv
if (operator.eq(argv[1], "64model")):
    input_size = 172
    #state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_64.model"
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_64_layer3.model"
elif (operator.eq(argv[1], "64diff")):
    input_size = 172
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_64diff.model"
elif (operator.eq(argv[1], "allin")):
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_allin.model"
else:
    state_dict_path = configs["state_dict_BLSTM_path"]
'''
if (operator.eq(argv[1], "None")):
    tarstatspath = configs["var_path"]
    state_dict_path = configs["state_dict_path"]
elif (operator.eq(argv[1], "diff")):
    tarstatspath = configs["var_diff_path"] # for diff
    state_dict_path = configs["state_dict_diff_path"] # for diff
elif (operator.eq(argv[1], "diffdnn3")):
    tarstatspath = configs["dnn3_var_diff_path"] # for diff dnn3
    state_dict_path = configs["state_dict_path2"] # for diff dnn3
'''
def train(model, trn_dl, device, optimizer, criterion):
    model.train()
    for i, (a, c) in enumerate(trn_dl):
        a, c = a.to(device), c.to(device)
        a.requires_grad_()
        optimizer.zero_grad()
        #print(a.shape)
        #print(c.shape)
        m_ = model.forward(a, None, False)

        loss = criterion(m_, th.squeeze(c))
        loss.backward()
        optimizer.step()

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

device = th.device("cuda" if use_cuda else "cpu")    

# read joint feature vector
if (operator.eq(argv[1], "nopad")):
    train_path = configs["train_blstmdata_path"]
    eval_path = configs["eval_blstmdata_path"]
    test_path = configs["test_blstmdata_path"]
elif (operator.eq(argv[1], "zeropad")):
    train_path = "./data/pair/SF1-TF1/jnt/dnn3_BLSTM_jntZ.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/dnn3_BLSTM_eval_jntZ.h5"
    test_path = "./data/pair/SF1-TF1/jnt/dnn3_BLSTM_test_jntZ.h5"
elif (operator.eq(argv[1], "diff")):
    train_path = configs["diff_train_blstmdata_path"]
    eval_path = configs["diff_eval_blstmdata_path"]
    test_path = configs["diff_test_blstmdata_path"]
elif (operator.eq(argv[1], "64model")):
    train_path = "./data/pair/SF1-TF1/jnt/blstm3_64_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/blstm3_64_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/blstm3_64_test_jnt.h5"
elif (operator.eq(argv[1], "64diff")):
    train_path = "./data/pair/SF1-TF1/jnt/blstm3_diff_64_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/blstm3_diff_64_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/blstm3_diff_64_test_jnt.h5"
elif (operator.eq(argv[1], "allin")):
    train_path = "./data/pair/SF1-TF1/jnt/blstm3_allin_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/blstm3_allin_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/blstm3_allin_test_jnt.h5"
elif(operator.eq(argv[1], "64fast")):
    input_size = 172
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_64fast.model"
    
    train_path = "./data/pair/SF1-TF1/jnt/blstm3_64fast_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/blstm3_64fast_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/blstm3_64fast_test_jnt.h5"
elif(operator.eq(argv[1], "64slow")):
    input_size = 172
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_64slow.model"
    
    train_path = "./data/pair/SF1-TF1/jnt/blstm3_64slow_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/blstm3_64slow_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/blstm3_64slow_test_jnt.h5"
elif(operator.eq(argv[1], "ap")):
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_ap2ap.model"
    
    train_path = "./data/pair/SF1-TF1/jnt/ap2ap_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/ap2ap_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/ap2ap_test_jnt.h5"
elif(operator.eq(argv[1], "onlyuv")):
    input_size = 176
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_onlyuv.model"
        
    train_path = "./data/pair/SF1-TF1/jnt/172_4_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/172_4_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/172_4_test_jnt.h5"
elif(operator.eq(argv[1], "apuv")):
    input_size = 216
    output_size = 80
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_apuv.model"
    
    train_path = "./data/pair/SF1-TF1/jnt/joint216_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/joint216_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/joint216_test_jnt.h5"
elif(operator.eq(argv[1], "ult_mcc")):
    input_size = 216
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_ult_mcc.model"
    
    train_path = "./data/pair/SF1-TF1/jnt/ult_mcc_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/ult_mcc_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/ult_mcc_test_jnt.h5"
elif(operator.eq(argv[1], "ult_apmcc")):
    input_size = 216
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_ult_apmcc.model"
    
    train_path = "./data/pair/SF1-TF1/jnt/ult_apmcc_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/ult_apmcc_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/ult_apmcc_test_jnt.h5"
elif(operator.eq(argv[1], "aplp")):
    input_size = 172
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_aplp.model"
    
    train_path = "./data/pair/SF1-TF1/jnt/aplp_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/aplp_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/aplp_test_jnt.h5"
    '''
    input_size = 120
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_ap_mcc.model"
    
    train_path = "./data/pair/SF1-TF1/jnt/ap160_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/ap160_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/ap160_test_jnt.h5"
    '''
elif(operator.eq(argv[1], "apmcc")):
    input_size = 80
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_apmcc.model"
    
    train_path = "./data/pair/SF1-TF1/jnt/mccap2ap_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/mccap2ap_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/mccap2ap_test_jnt.h5"
elif(operator.eq(argv[1], "64threedata")):
    #input_size = 172
    state_dict_path = "./data/pair/SF1-TF1/model/BLSTM_VC_three_layer3.model"
    num_layers = 3
    length_max = 512
    
    train_path = "./data/pair/SF1-TF1/jnt/blstm3_64slow_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/blstm3_64slow_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/blstm3_64slow_test_jnt.h5"
    train_file = HDF5(train_path, mode='r')
    train_data_slow = train_file.read(ext='mcep')
    train_file.close()
    print(train_data_slow.shape)
    eval_file = HDF5(eval_path, mode='r')
    eval_data_slow = eval_file.read(ext='mcep')
    eval_file.close()
    print(eval_data_slow.shape)
    test_file = HDF5(test_path, mode='r')
    test_data_slow = test_file.read(ext='mcep')
    test_file.close()
    print(test_data_slow.shape)
    
    train_path = "./data/pair/SF1-TF1/jnt/blstm3_64fast_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/blstm3_64fast_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/blstm3_64fast_test_jnt.h5"
    train_file = HDF5(train_path, mode='r')
    train_data_fast = train_file.read(ext='mcep')
    train_file.close()
    print(train_data_fast.shape)
    eval_file = HDF5(eval_path, mode='r')
    eval_data_fast = eval_file.read(ext='mcep')
    eval_file.close()
    print(eval_data_fast.shape)
    test_file = HDF5(test_path, mode='r')
    test_data_fast = test_file.read(ext='mcep')
    test_file.close()
    print(test_data_fast.shape)
    
    train_path = "./data/pair/SF1-TF1/jnt/blstm3_64_jnt.h5"
    eval_path = "./data/pair/SF1-TF1/jnt/blstm3_64_eval_jnt.h5"
    test_path = "./data/pair/SF1-TF1/jnt/blstm3_64_test_jnt.h5"
    train_file = HDF5(train_path, mode='r')
    train_data_nor = train_file.read(ext='mcep')
    train_file.close()
    print(train_data_nor.shape)
    eval_file = HDF5(eval_path, mode='r')
    eval_data_nor = eval_file.read(ext='mcep')
    eval_file.close()
    print(eval_data_nor.shape)
    test_file = HDF5(test_path, mode='r')
    test_data_nor = test_file.read(ext='mcep')
    test_file.close()
    print(test_data_nor.shape)
    
    train_data_ = np.r_[train_data_nor, train_data_fast]
    train_data_ = np.r_[train_data_, train_data_slow]
    eval_data_ = np.r_[eval_data_nor, eval_data_fast]
    eval_data_ = np.r_[eval_data_, eval_data_slow]
    test_data_ = np.r_[test_data_nor, test_data_fast]
    test_data_ = np.r_[test_data_, test_data_slow]
    print("total data size")
    print(train_data_.shape)
    print(eval_data_.shape)
    print(test_data_.shape)
    train_data_nor = None
    eval_data_nor = None
    test_data_nor = None
    train_data_fast = None
    eval_data_fast = None
    test_data_fast = None
    train_data_slow = None
    eval_data_slow = None
    test_data_slow = None

if(operator.eq(argv[1], "64threedata")):
    #pass
    
    # three_layer3
    train_data_ = np.c_[train_data_[:,:40], train_data_[:,-40:]]
    eval_data_ = np.c_[eval_data_[:,:40], eval_data_[:,-40:]]
    test_data_ = np.c_[test_data_[:,:40], test_data_[:,-40:]]
else:
    #train_path = configs["diff_train"]
    train_file = HDF5(train_path, mode='r')
    train_data_ = train_file.read(ext='mcep')
    length_max = 512 # train_file.read(ext='length_max')
    train_file.close()
    print("length = %d"%length_max)
    print(train_data_.shape)


    #eval_path = configs["diff_eval"]
    eval_file = HDF5(eval_path, mode='r')
    eval_data_ = eval_file.read(ext='mcep')
    eval_file.close()
    print(eval_data_.shape)


    #test_path = configs["diff_test"]
    test_file = HDF5(test_path, mode='r')
    test_data_ = test_file.read(ext='mcep')
    test_file.close()
    print(test_data_.shape)

# reshape data to 3d for LSTM input size
train_num = int(train_data_.shape[0]/length_max)
train_data_LSTM_ = np.reshape(train_data_, (train_num, length_max, input_size+output_size))
eval_num = int(eval_data_.shape[0]/length_max)
eval_data_LSTM_ = np.reshape(eval_data_, (eval_num, length_max, input_size+output_size))
test_num = int(test_data_.shape[0]/length_max)
test_data_LSTM_ = np.reshape(test_data_, (test_num, length_max, input_size+output_size))

train_data_LSTM = train_data_LSTM_
eval_data_LSTM = eval_data_LSTM_
test_data_LSTM = test_data_LSTM_
'''
train_data_LSTM = np.zeros([length_max, train_num, input_size*2])
for i in range(train_num):
    train_data_LSTM[:,i,:] = train_data_LSTM_[i,:,:]
eval_data_LSTM = np.zeros([length_max, eval_num, input_size*2])
for i in range(eval_num):
    eval_data_LSTM[:,i,:] = eval_data_LSTM_[i,:,:]
test_data_LSTM = np.zeros([length_max, test_num, input_size*2])
for i in range(test_num):
    test_data_LSTM[:,i,:] = test_data_LSTM_[i,:,:]
train_data_LSTM_ = None
eval_data_LSTM_ = None
test_data_LSTM_ = None
'''
#print(train_data_LSTM.shape)
#print(eval_data_LSTM.shape)
#print(test_data_LSTM.shape)

train_data_in_D = th.from_numpy(train_data_LSTM[:,:,:input_size]) # numpy->tensor
train_data_out_D = th.from_numpy(train_data_LSTM[:,:,input_size:])
eval_data_in_D = th.from_numpy(eval_data_LSTM[:,:,:input_size])
eval_data_out_D = th.from_numpy(eval_data_LSTM[:,:,input_size:])
test_data_in_D = th.from_numpy(test_data_LSTM[:,:,:input_size])
test_data_out_D = th.from_numpy(test_data_LSTM[:,:,input_size:])
#Double.tensor -> float.tensor
train_data_in = train_data_in_D.float() 
train_data_out = train_data_out_D.float()
eval_data_in = eval_data_in_D.float()
eval_data_out = eval_data_out_D.float()
test_data_in = test_data_in_D.float()
test_data_out = test_data_out_D.float()

train_data = data.TensorDataset(train_data_in, train_data_out)
eval_data = data.TensorDataset(eval_data_in, eval_data_out)
test_data = data.TensorDataset(test_data_in, test_data_out)

trn_dl_for_trainig = data.DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True )
trn_dl_for_eval = data.DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False )
val_dl = data.DataLoader(eval_data, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
tst_dl = data.DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=False)
# b = a.numpy() # tensor->numpy

# train dnn for mcep converted model
#variance = th.load(var_path)
model = dnnVC.BiRNN( input_size, hidden_size, output_size, num_layers).to(device)
optimizer = optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr = 0.0001 )
criterion = nn.MSELoss()
print("+==================================================================+")
print("|  Start Training Acoustic Model of the Source                     |")
print("+=========+===============+===============+===============+========+")
print("|  epoch  |   loss(trn)   |   loss(val)   |   loss(tst)   |  save  |")
print("+=========+===============+===============+===============+========+")
sys.stdout.flush()
min_trn_loss, min_val_loss, min_tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device, criterion)
print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(0, min_trn_loss, min_val_loss, min_tst_loss, " "))
print("+---------+---------------+---------------+---------------+--------+")
sys.stdout.flush()

for epoch in range(1, max_epoch+1):
    train(model, trn_dl_for_trainig, device, optimizer, criterion)
    trn_loss, val_loss, tst_loss = eval(model, trn_dl_for_eval, val_dl, tst_dl, device, criterion)
    save = False
    if val_loss < min_val_loss:
        save = True
        min_val_loss = val_loss
        th.save(model.state_dict(), state_dict_path)
        
    if save == True:
        print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(epoch, trn_loss, val_loss, tst_loss, "*"))
    else:
        print("|{:^9d}|{:15.5f}|{:15.5f}|{:15.5f}|{:^8}|".format(epoch, trn_loss, val_loss, tst_loss, " "))
        
    print("+---------+---------------+---------------+---------------+--------+")
    sys.stdout.flush()

# train dnn for ap(codeap / mcepap) converted model

# save model # read model ?



