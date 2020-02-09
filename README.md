# Open Comprehensive Blockchain Simulator

## Description
**OpenCBS** is a parallel computing project aiming at simulating distributed tasks
such as Bitcoin mining. This implementation employs a Message Passing Interface MPI approach
to distribute nonce computation jobs to workers. We enforce collective communication
throughout the mining by setting every worker to update the master once its finds a valid nonce,
or to be update once another node found a valid nonce. Note that batch split is also adopted as workers
operate on a given portion of the search range and gathering the results on the master
node which checks the existence of nonce and relaunches another batch if no valid nonce is obtained
from the previous batch, or initializes the mining of the next block if a valid nonce is obtained
for the current block.

In order to simulate the complexity of the target at every iteration/new block, multiplication and division
of the hexvalue of the strings is used as well as increasing the number of leading zeros. Hence, there is a noticeable
variation in the computation time for each block.

## Performance

For experimental purposes, a cluster of 10 nodes with CPU capability is used, and a total of 10 blocks are generated.
The figure below shows the time it took to find a valid nonce for each block.

![results](/opencbs/results.png)

*Fig 1. Runtime comparison*

Explanation of the graph: Get from model means that for every block, we compute the best chunk size from the model.
The second (default) means that we use a default value of 10^6. Note from the graph that as the number of leading
zeros increases getting from the model proves useful as using the default value increases the time faster.
We trained the model with different block headers as well as targets, currently forming a dataset of 397 unique records.
Adaboost and decision tree classifiers from Scikit-Learn are employed for training and inference, and the model with
the best accuracy is saved for our block generation task. The current best prediction accuracy is approximatel 64%.

**Please refer to the next section to understand the requirements of the system and the syntax for running experiments.**

## Instructions
### Packages
- Download and install JDK 8+
- Install python 2.7+ as well as packages such as numpy, pickle, 
    and pandas(if you would like to test the model generation program `Chunk_model.py`
- Download mpi.jar
- Generate a hostfile mentioning which machines you intend to use

### Syntax

- Compile with:
    - `mpijavac -cp <<path to mpi.jar>> opencbs.java MyTimer.java`
- Run with:
    - `mpirun --hostfile <<path to hostfile.txt>> --prefix /usr/local java -cp <<path to mpi.jar>> opencbs`

## Acknowledgement
Special thanks to *Dr Peizhao Hu* for his guidance, as well as key contribution leading to lines of codes in the java files contained in this repository.


