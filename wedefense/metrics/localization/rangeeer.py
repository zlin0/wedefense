#!/usr/bin/env python

# Copyright (c) 2025 Hieu-Thi Luong (contact@hieuthi.com)
# MIT License

"""
Fast calculating Range-based EER for spoof localization.
"""

import os
import sys
import argparse
import math
import numpy as np

from wedefense.utils.diarization.rttm_tool import get_rttm


def _pad_score_array(sco, lab):
	dur = lab[-1][-1]
	for i in range(len(sco)-1,0):
		if sco[i][1] <= dur:
			sco = sco[:i+1]
			break
	sco[-1][-1] = dur
	return sco

def _calculate_det_curve(counter):
	data = np.cumsum(counter, axis=1)
	data = np.divide(data, data[:,-2:-1])
	fpr = 1 - data[0,:]
	fnr = data[1,:]
	return fpr, fnr

def _calculate_eer(fpr, fnr):
	margin = np.abs(fpr - fnr)
	idxmin = np.argmin(margin)
	eer       = (fpr[idxmin]+fnr[idxmin])/2
	threshold = (idxmin+1) / margin.shape[0]
	return eer, threshold, margin[idxmin]

def _count_one_sample(counter, ref, hyp, resolution=8000, minval=-2.0, maxval=2.0):
	dur = ref[-1][-1]

	label = ref[0][0]                   # starting label
	rscur, recur, ridx = 0.0, 0.0, 0    # ref cursor
	hscur, hecur, hidx = 0.0, 0.0, 0    # hyp cursor

	totaldur = 0                        # for validating

	while rscur < dur:
		if rscur >= recur: # update reference if start touch end
			if ridx < len(ref): # there is annotation left
				ritem = ref[ridx]
				recur, ridx, label = ritem[2], ridx+1, ritem[0]
			else:
				break

		if hscur >= hecur: # update hypothesis if start touch end
			if hidx < len(hyp):
				hitem = hyp[hidx]
				hecur, hidx, score = hitem[2], hidx+1, hitem[0]
				score = (score-minval)/(maxval-minval)
			else:
				break

		# Create a predicted segment and update start cursors
		segdur = hecur-hscur if hecur<=recur else recur-rscur
		bidx   = int(score*resolution)
		counter[label,bidx] += segdur
		rscur = rscur+segdur
		hscur = rscur

		totaldur += segdur
		# DEBUG
		# print(f"split at {rscur} : {rscur} {recur} : {hscur} {hecur}")

	return counter

def _count_samples(labs, scos, resolution=8000, counter=None, minval=-2.0, maxval=2.0):
	if counter is None:
		counter = np.zeros((2,resolution+1))
	assert resolution+1 == counter.shape[1], "ERROR: the length of the preloaded counter and the resolution is not equal"
	names = labs.keys()
	for name in names:
		lab, sco = labs[name], scos[name]
		sco = _pad_score_array(sco, lab)
		counter = _count_one_sample(counter, lab, sco, resolution=resolution, minval=minval, maxval=maxval)
	return counter

def compute_rangeeer(labs, scos, resolution=8000, counter=None, minval=-2.0, maxval=2.0):
	counter  = _count_samples(labs, scos, resolution=resolution, counter=counter, minval=minval, maxval=maxval)
	fpr, fnr = _calculate_det_curve(counter)
	eer, threshold, margin = _calculate_eer(fpr,fnr)
	threshold = threshold * (maxval-minval) + minval
	return eer, threshold, margin, fpr, fnr, counter


def get_scores(score_file, score_index=1, score_duration=0.02, score_negative=False):
	scos, minscore, maxscore = {}, math.inf, -math.inf
	with open(score_file, 'r') as f:
		for idx, line in enumerate(f):
			args  = line.strip().split()
			name, i, score = args[0], int(args[1]), float(args[score_index])
			score = score if not score_negative else 1 - score
			item = [score, i*score_duration, (i+1)*(score_duration)]
			if name in scos:
				scos[name].append(item)
			else:
				scos[name] = [item]
			minscore = score if score < minscore else minscore
			maxscore = score if score > maxscore else maxscore
	return scos, minscore, maxscore

def normalize_labs(labs, label2id):
	newlabs = {}
	for name in labs.keys():
		lab=labs[name]
		newlab=[]
		for item in lab:
			newlab.append([label2id[item[0]], item[1], item[2]])
		newlabs[name] = newlab
	return newlabs

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Calculate Range EER')
	parser.add_argument('--score_file', type=str, help="Path to the score file.")
	parser.add_argument('--score_index', type=int, default=1, help="Index of the score value.")
	parser.add_argument('--score_negative', action="store_true", help="Score value if of the negative class")
	parser.add_argument('--rttm_file', type=str, help="Path to the ground-truth timestamp annotations file in the rttm format.")
	parser.add_argument('--frame_duration', type=float, default=0.02, help="Duration (s) of one frame.")
	parser.add_argument('--resolution', type=int, default=100000, help="Threshold resolution")
	args = parser.parse_args()

	labs, label2id = get_rttm(args.rttm_file)
	labs = normalize_labs(labs, label2id)
	scos, minscore, maxscore = get_scores(args.score_file, score_index=args.score_index, score_duration=args.frame_duration, score_negative=args.score_negative)
	maxval=max(abs(minscore),abs(maxscore))*2
	minval=-maxval

	eer, threshold, margin, fpr, fnr, counter = compute_rangeeer(labs, scos, resolution=args.resolution, minval=minval, maxval=maxval)
	print(f"rangeeer={eer*100:.3f}% margin={margin*100:.3f}% threshold={threshold:.3f} minscore={minscore:.3f} maxscore={maxscore:.3f} negative_class={args.score_negative}\n")
