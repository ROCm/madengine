#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#

cpus=""
if [ -f "/sys/fs/cgroup/cpuset/cpuset.cpus" ]; then
    cpus=$(cat /sys/fs/cgroup/cpuset/cpuset.cpus)
elif [ -f "/sys/fs/cgroup/cpuset.cpus.effective" ]; then
    cpus=$(cat /sys/fs/cgroup/cpuset.cpus.effective)
else
    echo "cpuset file is missing"
    exit 1
fi

cpus=${cpus//,/|}

echo 'model,performance,metric' > results_dummy_cpubind.csv
echo 'cpu,'${cpus}',cpu_bound' >> results_dummy_cpubind.csv
cp results_dummy_cpubind.csv ../
