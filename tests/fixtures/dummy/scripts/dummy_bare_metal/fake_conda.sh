#!/bin/bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Minimal fake conda for e2e-testing the bare-metal runner without a real
# conda/mamba install. Supports the subcommands CondaEnvManager invokes:
#   conda env list | conda env create | conda env update | conda env remove
#   conda create  | conda run -n <env> --no-capture-output <cmd...>
# `run` execs the wrapped command with CONDA_DEFAULT_ENV set to <env>.

case "$1" in
  env)
    case "$2" in
      list)
        echo "# conda environments"
        echo "base                  *  /tmp/fake_conda/base"
        ;;
      create|update) echo "fake conda env $2 ok" ;;
      remove) echo "fake conda env remove ok" ;;
    esac
    ;;
  create) echo "fake conda create ok" ;;
  run)
    shift
    env_name=""
    while [[ "$1" == -* || "$1" == "-n" ]]; do
      if [[ "$1" == "-n" ]]; then
        env_name="$2"
        shift 2
      else
        shift
      fi
    done
    export CONDA_DEFAULT_ENV="$env_name"
    exec "$@"
    ;;
  *) echo "fake conda: unknown $*" ;;
esac
