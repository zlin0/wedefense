#!/usr/bin/env python

# Copyright (c) 2025 Lin Zhang (partialspoof@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy as np
import argparse

from collections import defaultdict

###---------- to transfer vec to other labels. ----------
spf_detail_file="/home/smg/zhanglin/workspace/PROJ/02spf-dia/data/label/spf_detail.csv"
labidx={'input':1, 'inputpro':2, 'duration':3, 'conversion':4, 'spkrep':5, 'outputs':6, 'generator':7}
###---------- to transfer vec to other labels. ----------


def get_label2num(label2num_file):
    """
    Get the label2num based on the file. 

    Input: label2num file.

    Output: {label: num} 
        nonspeech: 0 
        A01: 1
    """

    label2num = dict( [line.split() for line in open(label2num_file) ] )
    num2label = {k:v for v,k in label2num.items()}

    return label2num, num2label



def get_rttm(rttm_file):
    """ Read rttm file and return in label-to-id mapping

    Args:
        rttm_file (str): Path to the RTTM file. Each line is expected to be:
        SPEAKER <reco_id> <channel> <start_time> <duration> <NA> <NA> <label> <NA> <NA>

    Returns:
        rttm (Dict[str, List[List[str, float, float]]])
             e.g., rttm = {
                  'rec1': [['bonafide', 0.0, 1.5], ['spoof', 1.5, 3.6]],
                  'rec2': [['bonafide', 0.1, 1.1]]
             }

        label2id: Dict[str, int]): Mapping each unique label to an integer ID.
    """
    
    rttm = defaultdict(list)
    label_set = []
    with open(rttm_file) as f:
        for line in f.readlines():
            _, reco , channels, st, dur, _, _, lab, _, _  = line.split() 
            if lab not in label_set:
                label_set.append(lab)
            start = float(st)
            end =  float(format(float(st) + float(dur), '.5f')) 
            rttm[reco].append([lab, st, end])
    label2id = {lab: idx for idx, lab in enumerate(label_set)}        
    return rttm, label2id


def get_vadvec(vad_file, seg_shift_sec):
    vad_matrix = np.loadtxt(vad_file)
    n_frames = int(max(vad_matrix[:,1]) / seg_shift_sec)
    vadvec = np.zeros(n_frames)
    for st, et, lab in vad_matrix:
        sf = int(round(st / seg_shift_sec))
        ef = int(round(et / seg_shift_sec))
        if (sf<ef):
            vadvec[sf : ef] = int(lab)
    return vadvec

def rttm2vadvec(rttm, seg_shift_sec, label2id, dur=0, skip_last=True):
    """
    Convert RTTM-style segments to frame-level label vector.

    Args:
        rttm (List[List[str, float, float]]): list of [label, start, end]
        shift_sec (float): frame duration in seconds
        label2id (Dict[str, int]): mapping from label name to ID
        dur (float): total duration of the utterance (optional)
        skip_last (bool): if True, floor the last frame

    Returns:
        np.ndarray: 1D array of label IDs per frame (int)

    """
    rttm = np.array(rttm)
    dur = float(dur)
    if(dur>0):
        pass
    else:
        dur = max(rttm[:,2])

    if(skip_last):
        n_frames = int(float(dur) / seg_shift_sec)
    else:
        n_frames = int(np.ceil(float(dur) / seg_shift_sec))

    vadvec = np.zeros(n_frames).astype(int)
    for lab, st, et in rttm:
        sf = max(0, int(st / seg_shift_sec + 0.5)) #To make sure at least one frame
        ef = max(sf + 1, int(et / seg_shift_sec + 0.5)) 
        ef = min(ef, n_frames)
        #sf = int(round(float(st) / seg_shift_sec))
        #ef = int(round(float(et) / seg_shift_sec))  
        vadvec[sf : ef] = label2id[lab]
    return vadvec

def vadvec2othlab(vadvec, othlab, othlab2num, label2num_file):

    label2num, num2label = get_label2num(label2num_file)
    newlab=[]
    #assert(type(vadvec)==np.ndarray)
    att2othlab=dict([[line.strip('\n').split(',')[0],line.strip('\n').split(',')[labidx[othlab]]] for line in open(spf_detail_file)])
    newlab = [othlab2num[att2othlab[num2label[i]]] for i in vadvec.astype(int).astype(str)]
        
    return np.array(newlab).astype(int)

#

def savepred_as_seg(args, res):
    #initial
    reco2dur = dict( [line.split() for line in open(args.reco2dur) ])

    seg_shift_sec = args.scale * 0.01
    seg_len_sec = seg_shift_sec

    f_segment = open(args.out_dir+'/res.segment', 'wt') #<seg> <reco> <st> <end>
    f_seglab = open(args.out_dir+'/res.seglab', 'wt')  #<seg> <lab>
    #finish initial
        
    if not isinstance(res, dict):
        res = res.tolist()

    for reco, pred in res.items():
        
        dur = float(reco2dur[reco])
        for idx, p in enumerate(pred):
            st =  idx * seg_shift_sec 
            end = st + seg_len_sec
            if(idx == len(pred)-1):
                end = dur
            segid = str(reco) + '_' + str(idx)
           
            f_segment.write('{} {} {:.4f} {:.4f}\n'.format(segid, reco, st, end))
            
            f_seglab.write('{} {}\n'.format(segid, str(int(p))))
        
    f_segment.close()
    f_seglab.close()
    
