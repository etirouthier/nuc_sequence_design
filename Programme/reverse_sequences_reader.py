#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:20:22 2019

@author: epierre
"""
import numpy as np
import random
from math import exp, sqrt
import matplotlib.pyplot as plt
from matplotlib import colors
import os, os.path
import argparse


from keras.models import load_model


from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mse_var, mae_cor


class Sequence:


    def __init__(self, folder):
        """
            Initiate a DNA sequence
            
            Args:
                length: the length of the artificial repeated element.
        """
        self.folder = folder
        self.number_files, self.length = self._get_number_and_size_of_files()
        self.sequence = self._load_sequences()
        #self.length = self._get_length(self.sequence)
        self.GC_content = self._get_gc_content_all(self.sequence)
        Sequence.WX = 2001

    @property
    def seq_predictable(self):
        """
            Change the sequence into a serie of one-hot-encoded window to serve
            as input of a model.
        """
        x_seq = self._process(self.sequence)
        return x_seq
    
    def _get_number_and_size_of_files(self):
        #print('debugg, folder:',self.folder)
        loc_fold = '../Results_nucleosome/'+self.folder
        fold_list = next(os.walk(loc_fold))[2]
        nb_files = len(fold_list)
        ini_seq = np.load(os.path.join(os.path.dirname(__file__),'..',
              'Results_nucleosome', self.folder, 
              'designed_sequence_0.npy'))
        lg = ini_seq.shape[0]
        print('debugg: nb_file:',nb_files)
        return nb_files, lg
    
    def _load_sequences(self):
        seq = np.empty([self.number_files, self.length])
        for i in range(0, self.number_files):
           seq[i,:] = np.load(os.path.join(os.path.dirname(__file__),'..',
              'Results_nucleosome', self.folder, 
              'designed_sequence_' + str(i) + '.npy'))
        #print('debugg, seq shape:', seq.shape)
        #x = np.concatenate(seq)
        #print('debugg, x shape:', x.shape)
        return seq #x
    
#    def _get_length(self, nucleotid):
#        print('debugg, sequence shape:',nucleotid.shape)
#        lg = nucleotid.shape[1]
#        return lg
        

    def _one_hot_encoder(self, nucleotid):
        res = (np.arange(nucleotid.max())==nucleotid[..., None]-1).astype(int)
        res = res.reshape(res.shape[0], 4)
        return res

    def _process(self, nucleotid):
        self.GC_content = self._get_gc_content_all(self.sequence)
        #print('shape GC content:',self.GC_content.shape)
        x_r = self._rescale(self.sequence)
        #print('shape after rescale:',x_r.shape)
        x_r = x_r.flatten()
        #print('shape after flatten:',x_r.shape)
        x_r = self._one_hot_encoder(x_r)
        #print('shape after one hot:',x_r.shape)
        HALF_WX = Sequence.WX // 2
        nw_seq_length = self.length + 2 * HALF_WX
        x_r = x_r.reshape(self.number_files,nw_seq_length,4)
        #print('shape after de-flatten:',x_r.shape)
        x_slide = rolling_window(x_r, window=(Sequence.WX, 4))
        #print('shape of x_slide:',x_slide.shape)
        x_seq = x_slide.reshape(x_slide.shape[0], 
                                x_slide.shape[1],Sequence.WX, 4, 1)
        #print('shape of x_seq:',x_seq.shape)
        x_seq = x_seq.reshape(self.number_files*self.length,Sequence.WX,4,1)
        #print('x_seq reshaped:',x_seq.shape)
        return x_seq
    
    def _rescale(self, nucleotid):
        HALF_WX = Sequence.WX // 2
        MARGIN = 2 * HALF_WX
        repeat_number = MARGIN // self.length + 3
        start = (HALF_WX // self.length + 1) * self.length - HALF_WX
        stop = start + self.length + 2 * HALF_WX
        new_sequence = np.tile(nucleotid, (1, repeat_number))
        return new_sequence[:,start : stop]
    
    def _get_gc_content_all(self, nucleotid):
        counter = np.zeros((self.number_files,1))
        #print('nucleotid:',nucleotid)
        for i in range(self.number_files):
            for j in range(self.length):
                if nucleotid[i,j]>2.5:
                    counter[i,0]+=1
            #print('sarace:',counter)
        counter/=self.length
        #print('GC content matrix shape:',counter.shape)
        return counter

class Energy:


    def __init__(self, seq, model, y_target):
        """
            This class is aimed at storing the previous energy so that to be
            able to reject a mutation.
            
            To initialise the energy we need the model on which the prediction
            will be made, the target function and also the first sequence (in a
            predictable shape). For representation purpose we also keep the
            predicted nucleosome density.
            
            Args:
                seq: numpy array of the one-hot-encoded initial sequence
                (with a rolling window applied so that prediction can be made).
                model: the trained keras model (could be multi-output)
                y_target: the target nucleosome density.
            Returns:
                energy: an Energy instance.
        """
        self.model = model
        self.y_target = y_target
        self.prediction = self.model.predict(seq.seq_predictable)
        self.prediction_profile = self._get_prediction_profile(self.prediction, seq)
        self.energy_profile = self._get_energy_profile(self.prediction_profile, seq)
       
    def _get_corr(self,y):
         X = y - np.mean(y)
         Y = self.y_target- np.mean(self.y_target)
         sigma_XY = np.sum(X*Y)
         sigma_X = np.sqrt(np.sum(X*X))
         sigma_Y = np.sqrt(np.sum(Y*Y))
         return sigma_XY/(sigma_X*sigma_Y + (7./3 - 4./3 - 1))#() is epsilon

    def _get_one_energy(self, y):
        one_energy = 5*np.mean(np.abs(y - self.y_target)) - \
               self._get_corr(y) + 1
        return one_energy
               
    def _get_energy(self, y):
        #print('shape of get_energy input:',y.shape)
        return np.mean([self._get_one_energy(y[:, i]) \
                        for i in range(y.shape[1])])
    
    def _get_prediction_profile(self, y, seq):
        #print('shape of prediction before split:',y.shape)
        y1 = seq.number_files
        y2 = seq.length
        return y.reshape(y1, y2, 1)
    
    def _get_energy_profile(self, y, seq):
        print('shape of prediction profile:',y.shape)
        coeff = 1.
        natural = 0.3806
        GC = seq.GC_content
        GC_energy = coeff*np.sqrt((GC-natural)*(GC-natural))
        GC_energy_1D= GC_energy.reshape((GC_energy.shape[0],))
        pred_energy = np.array([self._get_energy(y[i,:])\
                        for i in range(y.shape[0])])
        #print('shape of GC energy_1D', GC_energy_1D.shape)
        #print('shape of pred energy', pred_energy.shape)
        return pred_energy + GC_energy_1D



def _parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder',
                        help='''folder containing the artificial sequences''')
    parser.add_argument('-m','--model',
                        help='''Name of the model to be used for prediction''')
    return parser.parse_args(args)

def _sequence_changes(nucleotid, beginning_check, end_check):
    len_check = end_check - beginning_check
    #print('debugg: initial_seq shape:', nucleotid.sequence.shape)
    studied_seq = nucleotid.sequence[beginning_check:end_check, :]
    #print('debugg: studied_seq shape:', studied_seq.shape)
    changed_seq = np.copy(studied_seq)
    for i in range(0, len_check):
        for j in range(0, nucleotid.length):
            if studied_seq[i, j] == studied_seq[0, j]:
                changed_seq[i, j] = 1
            else:
                changed_seq[i, j] = 0
    return changed_seq

def main(command_line_arguments=None):
    """
        KMC algorithm to design a sequence that leads to a desired
        nucleosome density.
    """
    args = _parse_arguments(command_line_arguments)

    model = load_model(os.path.join(os.path.dirname(__file__),
                        '..',
                        'Results_nucleosome',
                        os.path.basename(os.path.abspath(args.model))),
                        custom_objects={'correlate': correlate,
                                        'mse_var': mse_var,
                                        'mae_cor': mae_cor})

    sequence = Sequence(args.folder)
    amp = 0.6 #0.3
    shift = 0.2 #0.2
    len601 = 147
    x_gauss = np.arange(1,len601,1)
    y_target_ = shift + \
        amp*np.exp(-((x_gauss - ((len601-1)/2.))**2/(2.*len601*len601/16.)))

    #y_target_ = np.append(y_target_, np.zeros((int(args.length) - (len601-1),))) 
    #linkers at zero
    y_target_ = np.append(y_target_, 
                          np.repeat(y_target_[-1],
                                    (int(sequence.length) - (len601-1)))) 
    #linkers at last value
    y_target = np.flip(y_target_)
    #y_target = y_target_

    energy = Energy(sequence, model, y_target) 
    #print(sequence.atgc_content)
    print('Initialization done!')
    print('Energy shape:', energy.energy_profile.shape)
    
    begin_s1 = 310
    end_s1 = 530
    changed_seq1 = _sequence_changes(sequence, begin_s1, end_s1)
    print('changed_seq shape', changed_seq1.shape)
    
    begin_s2 = 550
    end_s2 = 990
    changed_seq2 = _sequence_changes(sequence, begin_s2, end_s2)
    print('changed_seq shape', changed_seq2.shape)
    
    fig, (ax_energy, bx_seq) = plt.subplots(nrows=2, sharex=True)
    #fig = plt.figure(sharex=True)
    #ax_energy = fig.add_subplot(121)#121)
    plt.xlabel('Configuration')
    ax_energy.set_ylabel('Energy')
    ax_energy.plot(energy.energy_profile, 'r', label='energy')
    ax_energy.legend()
    #bx_seq = fig.add_subplot(122)
    
    # make a color map of fixed colors
    cmap = colors.ListedColormap(['red', 'green', 'orange', 'blue'])
    bounds=[0.5,1.5,2.5,3.5,4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # tell imshow about color map so that only set colors are used
    bx_seq.imshow(sequence.sequence.T, interpolation='nearest', origin='lower',
                    cmap=cmap, norm=norm, aspect='auto')
    
    #bx_seq.matshow(sequence.sequence, aspect='auto')
    bx_seq.set_ylabel('bp')
    bx_seq.legend()
    
    fig2, (ax_change1, ax_change2) = plt.subplots(ncols=2, sharey=True)
    ax_change1.imshow(changed_seq1.T, interpolation='nearest', origin='lower', aspect='auto')
    ax_change1.set_xlabel('configuration (transition to equilibrium)')
    ax_change1.set_ylabel('bp')
    ax_change1.legend()
    ax_change2.imshow(changed_seq2.T, interpolation='nearest', origin='lower', aspect='auto')
    ax_change2.set_xlabel('configuration (equilibrium)')
    ax_change2.set_ylabel('bp')
    ax_change2.legend()
    
    plt.show()
    
    '''
    store_nrj = list()
    counter = 0

    plt.ion()
    fig = plt.figure()
    ax_energy = fig.add_subplot(131)
    plt.xlabel('Configuration')
    plt.ylabel('Energy')
    ax_pred = fig.add_subplot(132)
    plt.xlabel('bp')
    plt.ylabel('Nucleosome occupency')
    ax_ATGC = fig.add_subplot(133)
    plt.xlabel('Configuration')
    plt.ylabel('A/T/G/C content')

    #plt.show()
    A_content = atgc[0,:]
    T_content = atgc[1,:]
    G_content = atgc[2,:]
    C_content = atgc[3,:]
    line_energy, = ax_energy.plot(range(counter), store_nrj)
    line_pred, = ax_pred.plot(y_target, 'r')
    mean_prediction = np.mean(energy.prediction_profile[0,:,:], axis=1)
    line_pred, = ax_pred.plot(mean_prediction, 'b')
    line_a, = ax_ATGC.plot(range(counter+1), A_content, 'r', label='A')
    line_t, = ax_ATGC.plot(range(counter+1), T_content, 'g', label='T')
    line_g, = ax_ATGC.plot(range(counter+1), G_content, 'tab:orange', label='G')
    line_c, = ax_ATGC.plot(range(counter+1), C_content, 'b', label='C')
    #plt.show()
    plt.legend()
    energy_file = open('energy.txt', 'w')

    for i in range(int(args.steps)):
        accepted = step(sequence, energy, float(args.temperature))
        sequence.sequence = sequence.mutated_seq[accepted,:]
        mean_prediction = np.mean(energy.prediction_profile[accepted,:,:], 
                                  axis=1)
        store_nrj.append(energy.energy_profile[accepted])
        s = str(i) + ', ' + str(energy.energy_profile[accepted])+'\n'
        energy_file.write(s)
        #atgc = np.append(atgc, sequence.atgc_content, axis=1)
        atgc = np.append(atgc, sequence._get_atgc_content_mutated(sequence.sequence), axis=1)
        #atgc = sequence._get_atgc_content_mutated(sequence.sequence)
        #print('sequence:',sequence.sequence)
        #print('A:',atgc[0,:])
        #print('T:',atgc[1,:])
        #print('G:',atgc[2,:])
        #print('C:',atgc[3,:])
        A_content = atgc[0,:].tolist()
        T_content = atgc[1,:].tolist()
        G_content = atgc[2,:].tolist()
        C_content = atgc[3,:].tolist()
        ATGC_content = A_content + T_content + G_content + C_content
        #print('A content:',A_content)
        #print('Stored energy:',store_nrj)

        if accepted:
            counter += 1
            print('nb of iteration:',counter)

        # saving the sequence with the minimal energy
        #if i > 1 and energy.energy_profile[accepted] < min(store_nrj[:-1]):
        #    np.save(os.path.join(os.path.dirname(__file__),
        #                         '..',
        #                         'Results_nucleosome',
        #                         'designed_sequence.npy'), sequence.sequence)
            
        np.save(os.path.join(os.path.dirname(__file__),
                                 '..',
                                 'Results_nucleosome',
                                 'designed_sequence_'+str(i)+'.npy'), sequence.sequence)      

        if i % 2 == 0:

            line_energy.set_ydata(store_nrj)
            line_energy.set_xdata(range(i + 1))
            ax_energy.set_ylim(-0.01, max(store_nrj) + 0.05)
            ax_energy.set_xlim(0, 1.8*i)
            
            line_pred.set_ydata(mean_prediction)
            ax_pred.set_ylim(min(np.min(mean_prediction),
                                 np.min(y_target)) - 0.05,
                             max(np.max(mean_prediction),
                                     np.max(y_target)) + 0.05)
            line_a.set_ydata(A_content)
            line_t.set_ydata(T_content)
            line_g.set_ydata(G_content)
            line_c.set_ydata(C_content)
            line_a.set_xdata(range(i + 2))
            line_t.set_xdata(range(i + 2))
            line_g.set_xdata(range(i + 2))
            line_c.set_xdata(range(i + 2))
            ax_ATGC.set_xlim(0, 1.8*i)
            ax_ATGC.set_ylim(min(ATGC_content)-0.01, max(ATGC_content)+0.01)

            fig.canvas.draw()
            ###fig.canvas.flush_events()
            plt.pause(1e-17)
            
        energy.compute_mutated_energy(sequence.seq_predictable)
        #print('energy profile:',energy.energy_profile)
    plt.ioff()
    plt.show()
    
    energy_file.close()
    '''
    
if __name__ == '__main__':
    main()
