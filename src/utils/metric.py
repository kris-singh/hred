#!/usr/bin/env python

def NLLLoss(gt_utt, pred_utt):
    gt_onehot = convert_to_onehot(gt_utt)
