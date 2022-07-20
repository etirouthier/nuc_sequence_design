#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:19:44 2019

@author: epierre
"""

import numpy as np
from math import exp, sqrt
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse


from keras.models import load_model


from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mse_var, mae_cor

def _parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    #parser.add_argument('-s', '--sequence', default="best_sequence_167_3_replica.fa",
    #                    help='''the artificial sequences''')
    parser.add_argument('-m','--model', default="weight_CNN_RNA_seq_2001_12_8_4_BY4742_rep01.hdf5",
                        help='''Name of the model to be used for prediction''')
    return parser.parse_args(args)

def _rescale(nucleotid, l):
    seq_WX = 2001
    HALF_WX = seq_WX // 2
    MARGIN = 2 * HALF_WX
    repeat_number = MARGIN // l + 1 + 2
    start = (HALF_WX // l + 1) * l - HALF_WX
    stop = start + 1 * l + 2 * HALF_WX
    new_sequence = np.tile(nucleotid, (1, repeat_number))
    return new_sequence[:,start : stop]

def _one_hot_encoder(nucleotid):
    res = (np.arange(nucleotid.max())==nucleotid[..., None]-1).astype(int)
    res = res.reshape(res.shape[0], 4)
    return res 

def _process(nucleotid, l):
    #print('shape nucleotide:',nucleotid.shape)
    seq_WX = 2001
    x_r = _rescale(nucleotid, l)
    x_r = x_r.flatten()
    x_r = _one_hot_encoder(x_r)
    HALF_WX = seq_WX // 2
    nw_seq_length = l + 2 * HALF_WX
    x_r = x_r.reshape(nw_seq_length,4)
    x_slide = rolling_window(x_r, window=(seq_WX, 4))
    x_seq = x_slide.reshape(x_slide.shape[0], 
                            x_slide.shape[1],seq_WX, 4, 1)
    x_seq = x_seq.reshape(l ,seq_WX,4,1)
    return x_seq

def _letter_to_numbers(nucleotid):
    L = len(nucleotid)
    seq = np.zeros(L)
    for i in range(L):
        if nucleotid[i] == "A": seq[i] = 1
        if nucleotid[i] == "T": seq[i] = 2
        if nucleotid[i] == "G": seq[i] = 3
        if nucleotid[i] == "C": seq[i] = 4
    return seq

def _get_reverse_complementary(nucleotid):
    complementary_nucleotid = np.copy(nucleotid)
    #print('debugg: shape nucleotid:', nucleotid.shape)
    for i in range(len(complementary_nucleotid)):
        complementary_nucleotid[i] = complementary_nucleotid[i] + 1 -\
        2*(complementary_nucleotid[i]//2) + 2*(complementary_nucleotid[i]//3)
    reverse_complementary_nucleotid = np.flip(complementary_nucleotid, 0)
    #print('debugg: rev shape nucleotid:', reverse_complementary_nucleotid.shape)
    whole_seq = np.concatenate((nucleotid, reverse_complementary_nucleotid))
    return whole_seq

def main(command_line_arguments=None):
    """
        test pred
    """
    
    #np.random.seed(2)#for debugg
    
    args = _parse_arguments(command_line_arguments)
    l = 197
    
    #197, 1 replica
    #test_seq_197 = "GCTCACTGCGAGCTGAATCATATCTTATTTAAGTCACTGATCATGACAGGATCCTCAATCGTTAAACCGGGCTACCATAGTGGGCCCTACTTTGATGCCCTTATAAGTCACTATCTTATGATGCTGAGAGTTAAACGTTTTGCAAATTTTCTTCGCGATTCTCGATGTAAGAGCGCAGCCCGGTTTGTTTGTAACCG"
    #197 3 replica v1
    #test_seq_197 = "AGAAAAACCACAGTATGGATTTATCTACAACAGTGAAATAAGTGAGAGTTCGACTGGACCAGTAAAAGTCAAGTTTGTGAGTCACCTATAGCATCTCAAGTTCGATATCTATTTATTTAGATGGCACTACGATGTTACTTCAAGTTTAAAAGCTAGCGGCTATACTCCGGCATGAGTTGCCAGGTTCTTTTTTGAGA"
    #197 3 replica v2
    test_seq_197 = "GAAGACTCTTCCGAGATAAGTTTTTTCCTTGATTGATAGATGAGTACGTCTGACCTATCTAGACCATGTGACTTCCGAGAGTGCTACATAGCTTCTACACATTCATCTATGAATCCTCAAATGGAAAGGCTGTATAAAAATGCCTTCGCGTGAATGTTTAGGCGTAAGGAGATTCGGTTGGCGCGCTCCGTGAAACA"
    
    test_seq_167 = "ATAGAAGCTAAATAAGTAGTATGAACTGATAAGGGACACAAGAGAACGAAGAACATTTCCTCCGCAACTTCGAATGCTGGTGTGTCTCTAGCGACGTTGGTCTAATGTGTTTGTACTTTTTTGGTCTTACGATGCTAGTATTGAGTTATAATTAATCGCCCAACAA"
    
    test_seq_237 = "GATTTACTTTAGAGCTAAACTATATATAGTGATAATTAATGTTGCAGCGAGGGCGTCCACGTATCCGAGCGACACCAGTGGGCACCGACTCTACTTCTCCCGTTCCCGTCAACGCTAATATCTTCATATTTCAAACCTATCTTTTAATGTATAAATCCCTTAGGCCCCGCCCTAATGCAGGTGGATCGAAAAAGCTTAAAATTCTGGCTTTGAGTAAAGGTTATAATGTAATTTTAA"
    
    num_seq = _letter_to_numbers(test_seq_197)
    
    rev_compl = _get_reverse_complementary(num_seq)
    
    proc_seq = _process(num_seq, l)
    rev_proc_seq = _process(rev_compl, l)
    
    #print('seq:', num_seq)

    model = load_model(os.path.join(os.path.dirname(__file__),
                        '..',
                        'Results_nucleosome',
                        os.path.basename(os.path.abspath(args.model))),
                        custom_objects={'correlate': correlate,
                                        'mse_var': mse_var,
                                        'mae_cor': mae_cor})
    
    prediction = model.predict(proc_seq)
    rev_pred = model.predict(rev_proc_seq)
        

    #print('predictions shape:', prediction.shape)
    #print(prediction)
    

    fig = plt.figure(figsize=(24, 13.5))
    ax_energy = fig.add_subplot(1,1,1)
    plt.xlabel('bp')#, fontsize=20)
    plt.ylabel('pred')#, fontsize=20)
    ax_energy.plot(np.mean(prediction[:,:], axis=1), marker='o', linestyle=' ', color='black')
    ax_energy.plot(np.mean(rev_pred[:,:], axis=1), marker='o', linestyle=' ', color='red')
    plt.show()

if __name__ == '__main__':
    main()