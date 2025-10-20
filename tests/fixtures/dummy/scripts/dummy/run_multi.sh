#!/bin/bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

echo "model,temperature,performance,metric
1,$RANDOM,$RANDOM,samples_per_sec
2,$RANDOM,$RANDOM,samples_per_sec
3,$RANDOM,$RANDOM,samples_per_sec
4,$RANDOM,$RANDOM,samples_per_sec" >>perf_dummy.csv

cp perf_dummy.csv ../
