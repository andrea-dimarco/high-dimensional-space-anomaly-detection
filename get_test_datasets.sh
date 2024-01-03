#!/bin/bash

# train
./generate_dataset n 2.0 2.0 10 1000 ./datasets/gaussian_0_1.csv

# test
./generate_dataset n 0.0 2.0 10 10000 ./datasets/gaussian_0_2.csv
./generate_dataset n 1.0 1.0 10 10000 ./datasets/gaussian_1_1.csv

./generate_dataset y 1.0 1.0 10 10000 ./datasets/uniform_1_1.csv
