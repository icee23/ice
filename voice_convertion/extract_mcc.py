#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract acoustic features for the speaker

"""

import argparse
import os
import sys
import json
import operator

from os import path

import numpy as np
import pysptk
from scipy.io import wavfile

from sprocket.speech import FeatureExtractor, Synthesizer
from sprocket.util import HDF5

from src.misc import low_cut_filter

argv = sys.argv

with open(path.join("configs","Configs_2.json")) as configs_file:
    configs = json.load(configs_file)

analyzer = configs["analyzer"]
fs = configs["sampling_rate"]
fftl = configs["wav_fftl"]
shiftms = configs["wav_shiftms"]
minf0 = configs["f0_minf0"]
maxf0 = configs["f0_maxf0"]
mgc_order=configs["mgc_order"]

# constract FeatureExtractor class
feat = FeatureExtractor(analyzer,
                            fs,
                            fftl,
                            shiftms,
                            minf0,
                            maxf0)

# constract Synthesizer class
synthesizer = Synthesizer(fs,
                              fftl,
                              shiftms)

#list_file = "/home/icee23/paper/20190312/data/CEMIX_word_16k.lst"
#list_file = "/home/icee23/paper/20190312/CEMIX_spell/CEMIX_spell.lst"
if(operator.eq(argv[1], "fast")):
    list_file = "/home/icee23/paper/20180919/sprocket-master/example/list/fast_lab_list.list"
    spath = "./data/wav/Fast_16k/"
elif(operator.eq(argv[1], "slow")):
    list_file = "/home/icee23/paper/20180919/sprocket-master/example/list/slow_lab_list.list"
    spath = "./data/wav/Slow_16k/"
elif(operator.eq(argv[1], "normal")):
    list_file = "/home/icee23/paper/20180919/sprocket-master/example/list/normal_lab_list.list"
    spath = "./data/wav/Normal_16k/"
    if(operator.eq(argv[2], "peak")):
        spath = "./data/wav/Normal_16k/Normal_peak/"
    elif(operator.eq(argv[2], "lound")):
        spath = "./data/wav/Normal_16k/Normal_lound/"
elif(operator.eq(argv[1], "median")):
    list_file = "/home/icee23/paper/20180919/sprocket-master/example/list/median_lab_list.list"
    spath = "./data/wav/Median_16k/"
    if(operator.eq(argv[2], "peak")):
        spath = "./data/wav/Median_16k/Median_peak/"
    elif(operator.eq(argv[2], "lound")):
        spath = "./data/wav/Median_16k/Median_lound/"
check_double = 0
doit = 0
# open list file
with open(list_file, 'r') as fp:
    for line in fp:
        f = line.rstrip()
        if(operator.eq(argv[2], "peak")):
            f = os.path.join(f + '_pn')
        elif(operator.eq(argv[2], "lound")):
            f = os.path.join(f + '_ln')
        h5f = os.path.join(spath+ f + '.h5')
        #h5f_2 = os.path.join(h5_dir, f + '_mcep.h5')
        if (check_double == 0):
            pass
        else: # check_double == 1
            if (not os.path.exists(h5f)):
                doit = 1
            else:
                doit = 0
                print("Acoustic features already exist: " + h5f)
        
        
        
        if(doit == 1 or check_double == 0):
            wavf = os.path.join(spath+ f + '.wav')
            fs, x = wavfile.read(wavf)
            x = np.array(x, dtype=np.float)
            x = low_cut_filter(x, fs, cutoff=70)

            print("Extract acoustic features: " + wavf)

            # analyze F0, spc, and ap
            f0, spc, ap = feat.analyze(x)
            mcep = feat.mcep(dim=mgc_order, alpha=configs["alpha_value"])
            npow = feat.npow()
            #codeap = feat.codeap()
            mcepap = pysptk.sp2mc(ap, mgc_order, configs["alpha_value"])
            #_ap = pysptk.mc2sp(mcepap, sconf.mcep_alpha, sconf.wav_fftl)
            #_spc = pysptk.mc2sp(mcep, sconf.mcep_alpha, sconf.wav_fftl)
            
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
            
            # save features into a hdf5 file
            h5 = HDF5(h5f, mode='a')
            h5.save(f0_rapt, ext='f0')
            #h5.save(spc, ext='spc')
            #h5.save(ap, ext='ap')
            h5.save(mcep, ext='mcep')
            h5.save(npow, ext='npow')
            #h5.save(codeap, ext='codeap')
            h5.save(mcepap, ext='mcepap')

            #h5.save(_spc, ext='spcmcep')
            #h5.save(_ap, ext='apmcep')
            h5.close()






