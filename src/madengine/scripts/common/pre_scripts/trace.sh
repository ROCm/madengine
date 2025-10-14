#!/usr/bin/env bash
# 
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
# 

set -e
set -x

os=''
if [ -n "$(command -v apt)" ]; then
	os=ubuntu
elif [ -n "$(command -v yum)" ]; then
	os=centos
else
	echo 'Unable to detect Host OS in pre_script'
	exit 1
fi

tool=$1

case "$tool" in

rpd)
	if [ "$os" == 'ubuntu' ]; then
		sudo apt update
		sudo apt install -y sqlite3 libsqlite3-dev libfmt-dev python3-pip
	elif [ "$os" == 'centos' ]; then
		sudo yum install -y libsqlite3x-devel.x86_64 fmt-devel python3-pip
	else
		echo "Unable to detect Host OS in trace pre-script"
	fi
	# Clone rocmProfileData repository
	if [ ! -d "rocmProfileData" ]; then
		git clone https://github.com/ROCm/rocmProfileData.git rocmProfileData
		if [ $? -ne 0 ]; then
			echo "Error: Failed to clone rocmProfileData repository"
			exit 1
		fi
	else
		echo "rocmProfileData directory already exists, skipping clone"
	fi
	
	# Build RPD tracer locally without system install
	cd ./rocmProfileData
	make rpd
	if [ $? -ne 0 ]; then
		echo "Error: Failed to build RPD tracer"
		exit 1
	fi
	
	# Install rocpd Python module locally
	cd rocpd_python
	python3 setup.py install
	if [ $? -ne 0 ]; then
		echo "Error: Failed to install rocpd Python module"
		exit 1
	fi
	cd ../..
	
	echo "RPD setup completed successfully"
	;;

esac
