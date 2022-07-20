#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:06:27 2019

@author: epierre
"""
import numpy as np
import random
from math import exp, sqrt
import matplotlib.pyplot as plt
import os
import argparse


from keras.models import load_model


from MyModuleLibrary.array_modifier import rolling_window
from MyModuleLibrary.mykeras.losses import correlate, mse_var, mae_cor


class Sequence:


    def __init__(self, length, repeat):
        """
            Initiate a DNA sequence
            
            Args:
                length: the length of the artificial repeated element.
                repeat: the number of repetition of the sequence on which to
                predict (1 is a good number...)
        """
        choices = [1, 2, 3, 4] #A, T, G, C
        '''roughly weights for singlet'''
        #weights = [0.3, 0.3, 0.2, 0.2]
        '''more accurate weights for singlets'''
        weights = [0.310052253746, 0.309303360736, 0.19039708206, 0.190247303458]
        '''pure random sequence 25% each'''
        #self.sequence = np.random.randint(1, 5, (length, 1))
        '''random sequence with weights on singlets'''
        #self.sequence = np.random.choice(choices, p=weights, size=(length, 1))
        '''exact random sequence with weights on singlets'''
        self.length = length
        self.sequence = self._get_ini_seq(self.length)
        self.mutated_seq = self._duplicate(self.sequence)
        self.repeat = repeat
        self.atgc_content = self._get_atgc_content(self.mutated_seq)
        Sequence.WX = 2001

    @property
    def seq_predictable(self):
        """
            Change the sequence into a serie of one-hot-encoded window to serve
            as input of a model.
        """
        x_seq = self._process(self.sequence)
        return x_seq

    @property
    def mutated_seq_predictable(self):
        """
            Change one nucleotid in the mutated_sequence and convert it into a
            predictable form.
        """   
        #self._propose_mutation()
        x_mutated_seq = self._process_mutated(self.mutated_seq)
        return x_mutated_seq
    
    def _get_ini_seq(self, length):
        choices = [1, 2, 3, 4] #A, T, G, C
        weights = [0.310052253746, 0.309303360736, 0.19039708206, 0.190247303458]
        nb_A = int(length*weights[0])
        nb_T = int(length*weights[1])
        nb_G = int(length*weights[2])
        nb_C = int(length*weights[3])
        seq = np.array([1] * nb_A + [2] * nb_T + [3] * nb_G + [4] * nb_C)
        while seq.size < length:
            seq = np.append(seq, np.random.choice(choices, p=weights))
        np.random.shuffle(seq)                   
        sequence = seq.reshape((length, 1))
        return sequence

    def _one_hot_encoder(self, nucleotid):
        res = (np.arange(nucleotid.max())==nucleotid[..., None]-1).astype(int)
        res = res.reshape(res.shape[0], 4)
        return res

    def _process(self, nucleotid):
        #print('shape nucleotide:',nucleotid.shape)
        x = self._duplicate(nucleotid)
        #print('shape after duplicate:',x.shape)
        self.mutated_seq = self._all_mutation(x)
        #print('shape after mutation:',self.mutated_seq.shape)
        x_r = self._rescale(self.mutated_seq)
        #print('shape after rescale:',x_r.shape)
        x_r = x_r.flatten()
        #print('shape after flatten:',x_r.shape)
        x_r = self._one_hot_encoder(x_r)
        #print('shape after one hot:',x_r.shape)
        HALF_WX = Sequence.WX // 2
        nw_seq_length = self.repeat * self.length + 2 * HALF_WX
        x_r = x_r.reshape(self.length-1,nw_seq_length,4)
        #print('shape after de-flatten:',x_r.shape)
        x_slide = rolling_window(x_r, window=(Sequence.WX, 4))
        #print('shape of x_slide:',x_slide.shape)
        x_seq = x_slide.reshape(x_slide.shape[0], 
                                x_slide.shape[1],Sequence.WX, 4, 1)
        #print('shape of x_seq:',x_seq.shape)
        x_seq = x_seq.reshape((self.length-1)*self.length,Sequence.WX,4,1)
        #print('x_seq reshaped:',x_seq.shape)
        return x_seq
    
    def _process_mutated(self, nucleotid):
        #print('shape nucleotide:',nucleotid.shape)
        self.mutated_seq = self._all_mutation(nucleotid)
        #print('shape after mutation:',self.mutated_seq.shape)
        x_r = self._rescale(self.mutated_seq)
        #print('shape after rescale:',x_r.shape)
        x_r = x_r.flatten()
        #print('shape after flatten:',x_r.shape)
        x_r = self._one_hot_encoder(x_r)
        #print('shape after one hot:',x_r.shape)
        x_r = x_r.reshape(self.length-1,2167,4)
        #print('shape after de-flatten:',x_r.shape)
        x_slide = rolling_window(x_r, window=(Sequence.WX, 4))
        #print('shape of x_slide:',x_slide.shape)
        x_seq = x_slide.reshape(x_slide.shape[0], 
                                x_slide.shape[1],Sequence.WX, 4, 1)
        #print('shape of x_seq:',x_seq.shape)
        x_seq = x_seq.reshape((self.length-1)*self.length,Sequence.WX,4,1)
        #print('x_seq reshaped:',x_seq.shape)
        return x_seq
    
    def _rescale(self, nucleotid):
        HALF_WX = Sequence.WX // 2
        MARGIN = 2 * HALF_WX
        repeat_number = MARGIN // self.length + self.repeat + 2
        start = (HALF_WX // self.length + 1) * self.length - HALF_WX
        stop = start + self.repeat * self.length + 2 * HALF_WX
        new_sequence = np.tile(nucleotid, (1, repeat_number))
        return new_sequence[:,start : stop]  

    ''' old mutation function     
    def _all_mutation(self, nucleotid):
        """mute 1 bp per line in a circular rotation 1 -> 2 -> 3 -> 4 -> 1"""
        for i in range(3*self.length):
            nucleotid[i, i//3] = nucleotid[i, i//3] + \
            1 + i%3 - 4 * ((nucleotid[i, i//3] + i%3) // 4)
        return nucleotid '''
    
    def _all_mutation(self, nucleotid):
        """permute 2 bp per line in order: 1<->2, 2<->3, 3<->4, ..."""
        perm_nucleotid = np.copy(nucleotid)
        for i in range(self.length-1):
            perm_nucleotid[i, i] = nucleotid[i, i+1]
            perm_nucleotid[i, i+1] = nucleotid[i, i]
        nucleotid = np.copy(perm_nucleotid)
        return nucleotid     

      
    def _duplicate(self, nucleotid):
        x_lin = np.ravel(nucleotid)
        x_dup = np.tile(x_lin, ((self.length-1), 1))
        return x_dup
        
    def _get_atgc_content(self, nucleotid):
        counter = np.zeros((4,1))
        #print('nucleotid:',nucleotid)
        for i in range(self.length):
            counter[nucleotid[0,i]-1,0]+=1
            #print('sarace:',counter)
        counter/=self.length
        #print('ATGC matrix:',counter)
        return counter
    
    def _get_atgc_content_mutated(self, nucleotid):
        counter = np.zeros((4,1))
        #print('nucleotid:',nucleotid)
        for i in range(self.length):
            counter[nucleotid[i]-1,0]+=1
            #print('sarace:',counter)
        counter/=self.length
        #print('ATGC matrix:',counter)
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
        self.prediction = self.model.predict(seq)
        self.prediction_profile = self._get_prediction_profile(self.prediction)
        self.energy_profile = self._get_energy_profile(self.prediction_profile)
        #self.mutated_pred = np.copy(self.prediction)
        #self.energy = self._get_energy(self.prediction)
        #self.mutated_energy = self._get_energy(self.mutated_pred)

#a virer
    def compute_mutated_energy(self, mutated_seq):
        """
            Calculate the energy of a mutated sequence and returns it.
            
            Args:
                mutated_seq: the one-hot-encoded proposed mutated sequence.
            Returns:
                Change self.mutated_energy to the new value.
        """
        self.mutated_pred = self.model.predict(mutated_seq)
        self.prediction_profile = self._get_prediction_profile(self.mutated_pred)
        self.energy_profile = self._get_energy_profile(self.prediction_profile)
        #self.mutated_energy = self._get_energy(self.mutated_pred)

        
    def _get_corr(self,y):
         X = y - np.mean(y)
         Y = self.y_target- np.mean(self.y_target)
         sigma_XY = np.sum(X*Y)
         sigma_X = np.sqrt(np.sum(X*X))
         sigma_Y = np.sqrt(np.sum(Y*Y))
         return sigma_XY/(sigma_X*sigma_Y + (7./3 - 4./3 - 1))#() is epsilon

    def _get_one_energy(self, y):
        #print('shape of get_one_energy input:',y.shape)
        return 5*np.mean(np.abs(y - self.y_target)) - \
               self._get_corr(y) + 1  #added weight on mae
               #np.corrcoef(y, self.y_target)[0, 1] + 1  #added weight on mae
               
    def _get_energy(self, y):
        #print('shape of get_energy input:',y.shape)
        return np.mean([self._get_one_energy(y[:, i]) \
                        for i in range(y.shape[1])])
    
    def _get_prediction_profile(self, y):
        #print('shape of prediction before split:',y.shape)
        dim = 167 #A FAIRE PROPRE!!!!
        #y1 = int(3*dim)
        y1 = int(dim-1)
        y2 = int(dim)
        return y.reshape(y1, y2, 1)
    
    def _get_energy_profile(self, y):
        #print('shape of prediction profile:',y.shape)
        return np.array([self._get_energy(y[i,:])\
                        for i in range(y.shape[0])])


def step(sequence, energy, temp):
    """
        Propose a mutation of the sequence and accept the best mutation
        on the basis of Kinetic Monte Carlo algorithm.
        
        We propose a mutation and calculate the predicted density of nucleosome
        on the mutated sequence. The energy is the euclidian distance between
        the density predict and the density that we finally want to have.
        
        Args:
            sequence: a Sequence instance
            energy: Energy instances corresponding to the sequence
            temp: the temperature of the simulation
        Return:
            index: the index of the selected mutation
    """
    #energy.compute_mutated_energy(sequence.mutated_seq_predictable)
    #print('energy profile:',energy.energy_profile)
    esum = np.sum(np.exp(-(energy.energy_profile)/temp))
    #print('esum:',esum)
    rand = random.uniform(0,esum)
    #print('rand:',rand)
    index = 0
    eless = 0
    while eless <= rand:
        #print('eless:',eless)
        eless = eless + exp(-(energy.energy_profile[index])/temp)
        index = index + 1
    print('chosen index:',index-1)
    return index-1

def _parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--length',
                        help='''the length of the artificial sequence''')
    parser.add_argument('--repeat',
                        help='''the number of artificial sequence we want to
                        predict on''')
    parser.add_argument('-s','--steps',
                        help='''Number of KMC iterations.''')
    parser.add_argument('-m','--model',
                        help='''Name of the model to be used for prediction''')
    parser.add_argument('-t', '--temperature',
                        help='''temperature used in the KMC algorithm''')
    return parser.parse_args(args)

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
                                    (int(args.length) - (len601-1)))) 
    #linkers at last value
    y_target = y_target_
    for i in range(int(args.repeat) - 1):
        y_target = np.append(y_target, y_target_)

    sequence = Sequence(int(args.length), int(args.repeat))
    energy = Energy(sequence.seq_predictable, model, y_target) 
    #print(sequence.atgc_content)
    print('Initialization done!')
    atgc = sequence.atgc_content
    store_nrj = list()
    counter = 0
    
    #print('A:',atgc[0,:])
    #print('T:',atgc[1,:])
    #print('G:',atgc[2,:])
    #print('C:',atgc[3,:])

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

    for i in range(int(args.steps)):
        accepted = step(sequence, energy, float(args.temperature))
        sequence.sequence = sequence.mutated_seq[accepted,:]
        mean_prediction = np.mean(energy.prediction_profile[accepted,:,:], 
                                  axis=1)
        store_nrj.append(energy.energy_profile[accepted])
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
        if i > 1 and energy.energy_profile[accepted] < min(store_nrj[:-1]):
            np.save(os.path.join(os.path.dirname(__file__),
                                 '..',
                                 'Results_nucleosome',
                                 'designed_sequence.npy'), sequence.sequence)

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

if __name__ == '__main__':
    main()
