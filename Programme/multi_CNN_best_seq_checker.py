#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:45:21 2019

@author: epierre
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        self.number_files = self._get_number_of_files()
        self.length, self.repeat, self.step, self.model, self.temp,\
        self.directory, self.weight_GC, self.weight_seq, self.weight_rev,\
        self.weight_mut, self.target_amp, self.target_bk = self._get_attributes()
        self.min_e_seq = self._get_min_energy_sequence()
        self.sequence = self._load_sequence()
        self.A_content, self.T_content, self.G_content,\
        self.C_content = self._get_atgc_content(self.sequence)
        self._save_best_sequence_as_npy(self.sequence)
        self._save_best_sequence_as_fasta(self.sequence)
        Sequence.WX = 2001

    @property
    def seq_predictable(self):
        """
            Change the sequence into a serie of one-hot-encoded window to serve
            as input of a model.
        """
        x_seq = self._process(self.sequence)
        return x_seq
    
    def _get_number_of_files(self):
        #print('debugg, folder:',self.folder)
        loc_fold = '../Results_nucleosome/'+self.folder
        fold_list = next(os.walk(loc_fold))[2]
        nb_files = len(fold_list)
        print('Number of steps done:', nb_files)
        return nb_files
    
    def _load_sequence(self):
        seq= np.load(os.path.join(os.path.dirname(__file__),'..',
              'Results_nucleosome', self.folder, 
              'designed_sequence_' + str(self.min_e_seq) + '.npy'))
        print('sequence:',seq)
        return seq 
    
    def _get_attributes(self):
        #colnames=['length','repeat','step','model','temperature	','directory',
        #          'weight_GC','weight_seq','weight_rev_comp	','weight_mut',
        #          'target_amplitude	','target_background']
        df = pd.read_csv(os.path.join(os.path.dirname(__file__),'..',
              'Results_nucleosome', self.folder, 'config.txt'), sep='\t')#, names=colnames)
        lg = int(df['length'])
        repe = int(df['repeat'])
        stp = int(df['step'])
        mod = str(df.model.values[0])
        tps = float(df['temperature'])
        dire = str(df.directory.values[0])
        wGC = float(df['weight_GC'])
        wseq = float(df['weight_seq'])
        wrev = float(df['weight_rev_comp'])
        wmut = float(df['weight_mut'])
        tamp = float(df['target_amplitude'])
        tbk = float(df['target_background'])
        print('config file loaded')
        print('length:', lg)
        print('nb of repeat:', repe)
        print('nb of step:', stp)
        print('model used:', mod)
        print('temperature:', tps)
        print('directory:', dire)
        print('weight GC:', wGC)
        print('weight sequence:', wseq)
        print('weight reverse complement:', wrev)
        print('weight done mutations:', wmut)
        print('target amplitude:', tamp)
        print('target background:', tbk)
        return lg, repe, stp, mod, tps, dire, wGC, wseq, wrev, wmut, tamp, tbk
    
    def _get_min_energy_sequence(self):
        colnames=['config', 'energy']
        df = pd.read_csv(os.path.join(os.path.dirname(__file__),'..',
              'Results_nucleosome', self.folder, 'energy.txt'), names=colnames,
               header=None)
        min_e = df['energy'].min()
        print('minimal energy:', min_e)
        min_col = df.iloc[df['energy'].argmin()]
        min_config = int(min_col['config'])
        print('best config:', min_config)
        return min_config
        

    def _one_hot_encoder(self, nucleotid):
        res = (np.arange(nucleotid.max())==nucleotid[..., None]-1).astype(int)
        res = res.reshape(res.shape[0], 4)
        return res

    def _process(self, nucleotid):
        x_r = self._rescale(self.sequence)
        #print('shape after rescale:',x_r.shape)
        x_rc = self._get_reverse_complementary(x_r)
        x_rc = x_rc.flatten()
        #print('shape after flatten:',x_r.shape)
        x_rc = self._one_hot_encoder(x_rc)
        #print('shape after one hot:',x_r.shape)
        HALF_WX = Sequence.WX // 2
        nw_seq_length = self.length + 2 * HALF_WX
        x_rc = x_rc.reshape(2,nw_seq_length,4)
        #print('shape after de-flatten:',x_r.shape)
        x_slide = rolling_window(x_rc, window=(Sequence.WX, 4))
        #print('shape of x_slide:',x_slide.shape)
        x_seq = x_slide.reshape(x_slide.shape[0], 
                                x_slide.shape[1],Sequence.WX, 4, 1)
        #print('shape of x_seq:',x_seq.shape)
        x_seq = x_seq.reshape(2*self.length,Sequence.WX,4,1)
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
    
    def _get_reverse_complementary(self, nucleotid):
        complementary_nucleotid = np.copy(nucleotid)
        #print('debugg: shape nucleotid:', nucleotid.shape)
        for i in range(len(complementary_nucleotid)):
            complementary_nucleotid[i] = complementary_nucleotid[i] + 1 -\
            2*(complementary_nucleotid[i]//2) + 2*(complementary_nucleotid[i]//3)
        reverse_complementary_nucleotid = np.flip(complementary_nucleotid, 1)
        #print('debugg: rev shape nucleotid:', reverse_complementary_nucleotid.shape)
        whole_seq = np.concatenate((nucleotid, reverse_complementary_nucleotid))
        return whole_seq

    def _get_atgc_content(self, nucleotid):
        counter_A = 0
        counter_T = 0
        counter_G = 0
        counter_C = 0
        #print('nucleotid:',nucleotid)
        for i in range(self.length):
            if nucleotid[i] == 1: counter_A = counter_A + 1
            if nucleotid[i] == 2: counter_T = counter_T + 1
            if nucleotid[i] == 3: counter_G = counter_G + 1
            if nucleotid[i] == 4: counter_C = counter_C + 1
        return counter_A, counter_T, counter_G, counter_C
    
    def _save_best_sequence_as_npy(self, nucleotid):
        np.save(os.path.join(os.path.dirname(__file__),
                                 '..', 'Results_nucleosome', 
                                 'best_sequence_'+str(self.length)+'.npy'),
            nucleotid)
    
    def _save_best_sequence_as_fasta(self, nucleotid):
        transition = []
        for i in range(len(nucleotid)):
            if nucleotid[i] == 1: transition.append('A')
            if nucleotid[i] == 2: transition.append('T')
            if nucleotid[i] == 3: transition.append('G')
            if nucleotid[i] == 4: transition.append('C')
        with open(os.path.join(os.path.dirname(__file__),
                                 '..', 'Results_nucleosome',
                                 'best_sequence_'+str(self.length)+'_3_replica.fa'), 'w') as fasta:
            fasta.write('>best_sequence%s\n' % (str(self.length)))
            for i in range(len(transition)):
                fasta.write(transition[i])
                if (i+1) % 50 == 0 : fasta.write('\n')

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
    
    def _get_prediction_profile(self, y, seq):
        #print('shape of prediction before split:',y.shape)
        y1 = 2
        y2 = seq.length
        return y.reshape(y1, y2, 1)
    
def _parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default="test",
                        help='''directory containing the artificial sequences''')
    parser.add_argument('-m','--model', default="weights_3_replica_v2.hdf5",
                        help='''Name of the model to be used for prediction''')
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

    sequence = Sequence(args.directory)
    
    amp = sequence.target_amp
    shift = sequence.target_bk
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
    y_target = y_target_

    energy = Energy(sequence, model, y_target) 
    
    other_models = ['weights_WT_167_chd1_rep1.hdf5',
                    'weights_WT_167_chd1_rep2.hdf5',
                    'weights_WT_167_rep2.hdf5',
                    'weights_WT_167_rep3.hdf5',
                    'weights_WT_197_chd1_rep1.hdf5',
                    'weights_WT_197_chd1_rep2.hdf5',
                    'weights_WT_237_rep2.hdf5']
    
    length_mod = len(other_models) 
    i = 1
    dm={}
    de={}
    dp={}
    drp={}
    while i < length_mod + 1: 
        print('Now computing model', other_models[i-1]) 
        dm["model{0}".format(i)] = load_model(os.path.join(os.path.dirname(__file__),
                        '..',
                        'Results_nucleosome',
                        os.path.basename(os.path.abspath(other_models[i-1]))),
                        custom_objects={'correlate': correlate,
                                        'mse_var': mse_var,
                                        'mae_cor': mae_cor})
        de["energy{0}".format(i)] = Energy(sequence, dm["model{0}".format(i)], y_target)
        dp["mean_prediction{0}".format(i)] = np.mean(de["energy{0}".format(i)].prediction_profile[0,:,:], axis=1)
        drp["mean_rev_prediction{0}".format(i)] = np.flip(np.mean(de["energy{0}".format(i)].prediction_profile[1,:,:], axis=1), 0)
        i += 1
    
    mean_prediction = np.mean(energy.prediction_profile[0,:,:], axis=1)
    mean_com_prediction = np.mean(energy.prediction_profile[1,:,:], axis=1)
    mean_rev_prediction = np.flip(mean_com_prediction, 0)
    
    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(20)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(0, length_mod + 1):
        ax = fig.add_subplot(2, 4, i+1)
        plt.xlabel('bp')
        plt.ylabel('Nucleosome occupency')
        plt.title('reference CNN') if i == 0 else plt.title(other_models[i-1])
        ax.plot(y_target, 'r', label='target')
        ax.plot(mean_prediction, 'b', label='pred') if i == 0 else ax.plot(dp["mean_prediction{0}".format(i)], 'b', label='pred')
        ax.plot(mean_rev_prediction, 'k', label='rev_com_pred') if i == 0 else ax.plot(drp["mean_rev_prediction{0}".format(i)], 'k', label='rev_com_pred')
        ax.legend()
    
    
    
    
    
    
    #ax_pred = fig.add_subplot(1,1,1)
    #plt.xlabel('bp')
    #plt.ylabel('Nucleosome occupency')
    #plt.title('reference CNN')
    #line_pred, = ax_pred.plot(y_target, 'r', label='target')
    #line_pred_reg, = ax_pred.plot(mean_prediction, 'b', label='pred')
    #line_pred_rev, = ax_pred.plot(mean_rev_prediction, 'k', label='rev_com_pred')
    #ax_pred.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__),
                                 '..','Results_nucleosome',
                                 'best_config_'+str(sequence.length)+'_3_replica.png'))
    plt.show()
    
    """
    #print(sequence.atgc_content)
    print('Initialization done!')
    print('Energy shape:', energy.energy_profile.shape)
    
    begin_s1 = 1
    end_s1 = 101
    changed_seq1 = _sequence_changes(sequence, begin_s1, end_s1)
    print('changed_seq shape', changed_seq1.shape)
    
    begin_s2 = 1
    end_s2 = 101
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
    """
    
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
