#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:22:35 2019

@author: epierre
"""

import numpy as np


def duplicate(nucleotid):
    x_lin = np.ravel(nucleotid)
    x_dup = np.tile(x_lin, (3*(len(nucleotid)), 1))
    return x_dup

def all_mutation(nucleotid):
    """mute 1 bp per line in a circular rotation 1 -> 2 -> 3 -> 4 -> 1"""
    for i in range(len(nucleotid)):
        nucleotid[i, i//3] = nucleotid[i, i//3] + \
        1 + i%3 - 4 * ((nucleotid[i, i//3] + i%3) // 4)
    return nucleotid 

def get_reverse_complementary(nucleotid):
    complementary_nucleotid = np.copy(nucleotid)
    for i in range(len(complementary_nucleotid)):
        complementary_nucleotid[i] = complementary_nucleotid[i] + 1 -\
        2*(complementary_nucleotid[i]//2) + 2*(complementary_nucleotid[i]//3)
    reverse_complementary_nucleotid = np.flip(complementary_nucleotid, 1)
    #print('debugg: rev shape nucleotid:', reverse_complementary_nucleotid.shape)
    whole_seq = np.concatenate((nucleotid, reverse_complementary_nucleotid))
    return whole_seq

def main():
    x = np.array([[1],[2],[3]])
    print('Initial sequence shape:',x.shape)
    print('Initial sequence:',x)
    y = duplicate(x)
    print('Shape after duplication',y.shape)
    print('Sequence after duplication:',y)
    z = all_mutation(y)
    print('Shape after mutation',z.shape)
    print('Sequence after mutation:',z)
    seq = get_reverse_complementary(z)
    print('Shape with rev compl',seq.shape)
    print('Sequence with rev compl:',seq)
    
if __name__ == '__main__':
    main()