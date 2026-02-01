#!/bin/bash

# Activate virtual environment
source env/bin/activate
nohup ./run_all_combinations.sh > run7.log 2>&1 &
wait