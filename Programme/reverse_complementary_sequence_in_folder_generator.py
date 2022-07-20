#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:12:41 2019

@author: epierre
"""

import numpy as np
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory',
                        help = '''directory containing the DNA sequence''')
    return parser.parse_args()

def make_output_folder(folder):
    outfolder = folder + '_reverse'
    path_to_new_folder = os.path.join('..', 'Results_nucleosome', outfolder)
    try:
    # Create target Directory
        os.mkdir(path_to_new_folder)
        print("Directory " , outfolder,  " Created ") 
    except FileExistsError:
        print("Directory " , outfolder,  " already exists")

def get_number_of_files(folder):
    #print('debugg, folder:',folder)
    loc_fold = '../Results_nucleosome/'+folder
    fold_list = next(os.walk(loc_fold))[2]
    nb_files = len(fold_list)
    #print('number of files:',nb_files)
    return nb_files

def load_data(file_number, folder):
    path_to_input_file = os.path.join(os.path.dirname(__file__),'..',
              'Results_nucleosome', folder, 
              'designed_sequence_' + str(file_number) + '.npy')
    nucleotid = np.load(path_to_input_file)
    print('done!')
    return nucleotid

def get_reverse_complementary(nucleotid):
    complementary_nucleotid = nucleotid
    for i in range(len(complementary_nucleotid)):
        complementary_nucleotid[i] = complementary_nucleotid[i] + 1 -\
        2*(complementary_nucleotid[i]//2) + 2*(complementary_nucleotid[i]//3)
    reverse_complementary_nucleotid = np.flip(complementary_nucleotid, 0)
    return reverse_complementary_nucleotid

def save_data(nucleotid, file_number, folder):
    outfolder = folder + '_reverse'
    path_to_output_file = os.path.join(os.path.dirname(__file__),'..',
              'Results_nucleosome', outfolder, 
              'designed_sequence_' + str(file_number) + '.npy')
    
    np.save(path_to_output_file, nucleotid)
    print('File', file_number, 'saved!')
    
def main():
    args = parse_arguments()
    folder = args.directory
    make_output_folder(folder)
    nb_input_files = get_number_of_files(folder)
    for i in range(0, nb_input_files):
        data = load_data(i, folder)
        #print(' input data:', data)
        reverse_complementary_data = get_reverse_complementary(data)
        #print(' output data:', reverse_complementary_data)
        save_data(reverse_complementary_data, i, folder)

if __name__ == '__main__':
    main()