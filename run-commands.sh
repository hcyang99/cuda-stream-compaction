#!/bin/sh
# nvcc --help
# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

# Recompile if necessary (DO NOT CHANGE!)
mkdir -p build
cd build
cmake ..
make -j4 && ./scan

## Basic run to map the first 40 reads in the reads.fa in batches of 10 reads
## HINT: may need to change values for the assignment tasks. You can create a sequence of commands
