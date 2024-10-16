#!/bin/bash

#Slurm sbatch options
#SBATCH -o runNLP_RNN_new.log_spec%j
#SBATCH --job-name pytorch
#SBATCH -n 1
# SBATCH -c 40
#SBATCH --exclusive
# SBATCH -N 1
# SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:volta:2
# Load Anaconda and MPI module
module load anaconda/2023a
module load mpi/openmpi-4.1.5
#module load cuda/11.6
#module load nccl/2.11.4-cuda11.6

# Call your script as you would from the command line
#mpirun python clusteringKMeans_score.py #set n = 15
#mpirun python clusteringKMeans.py
#mpirun python clusteringGauss_score.py #set n = 15
#mpirun ${MPI_FLAGS} python clusteringSpectral_score.py
#mpirun python clusteringSpectral_score.py
#mpirun python RNN_initial.py # set n=1 but c to 20 
#mpirun python RNN.py
#mpirun python RNN_model_from_file.py
#mpirun python TxtClass.py
#mpirun python MyModel.py
#mpirun python run_spacy.py
#mpirun python run_spacy_small.py
#mpirun python run_spacy_small2.py
#mpirun python run_spacy_df_codes.py
#mpirun python RFClass.py
#mpirun python Fuzzy.py
#mpirun python RNN_read.py
#mpirun python RNN-em.py
mpirun python Final.py


