# nuc_sequence_design

## Description:

Use this repository to design a DNA sequence that attract nucleosome in yeast using the kinetic Monte-Carlo methodology (KMC) ! 

The sequence of length 167 bp, 197 bp or 237 bp will evolved step by step through a nucleosome positioning sequence (i.e a sequence where a nucleosome is set on the first 147 bp). At every step the algorithm will choose a mutation to perform that tends to favor the positioning power of the sequence. To do so it uses internally a deep learning model able to predict the nucleosome density associated with any sequence in yeast.

------------------


## Getting started:

To design a sequence use the module in `Programme`especially the last one `sequence_design_all_mut_w_rev_and_weights_w_repeat.py`.

The command are as follows :
  - -l or --length : length of the sequence (default is 167)
  - -r or --repeat : number of repeat to predict on (default is 1)
  - -s or --steps : number of KMC iterations (default is 10000)
  - -m or --model : a hdf5 file containing the model to predict with
  - -t or --temperature : the "temperature" to use in the model (default is 0.1)
  - -d or --directory : the output directory
  - -w or --weights : the weights used in the energy function [a, b, c, d] with a for the CG content energy, b for the positioning energy in the direct strand, c for the energy in the reverse strand and d for the mutation energy (default [1, 1, 1, 1])
  - -g or --gauss : amplitude and background of the gaussian target (default [0.4, 0.2])
  
  ------------------


## Results:

Two directory are created with the specified name in `Results_nucleosome`.

The first one contains a config.txt file to store the chosen configuration and a energy.txt file with the energy at every steps of the KMC optimisation. The second directory stores the sequence at every steps of the KMC optimization and several plots.
