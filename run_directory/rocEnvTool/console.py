#!/usr/bin/env python3
"""Module to run console commands

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
import subprocess

class Console:
  """ Console class
  class to run console commands
  """
  def __init__(self, shellVerbose=True, live_output=False):
      self.shellVerbose = shellVerbose
      self.live_output = live_output

  def sh(self, command, canFail=False, timeout=60, secret=False, prefix=""):
      if self.shellVerbose and not secret:
          print("> " + command, flush=True)
      proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, universal_newlines=True, bufsize=1)
      try:
          if not self.live_output: 
              outs, errs = proc.communicate(timeout=timeout)
          else:
              outs = []
              for stdout_line in iter(proc.stdout.readline, ""):
                  print(prefix+stdout_line, end="" )
                  outs.append(stdout_line )
              outs = ''.join(outs)
              proc.stdout.close()
              proc.wait(timeout=timeout) 
      except subprocess.TimeoutExpired as exc:
          proc.kill()
          raise RuntimeError('Console script timeout') from exc
      if proc.returncode != 0:
          if not canFail:
            if not secret:
                raise RuntimeError("Subprocess '" + command + "' failed with exit code " + str(proc.returncode) )
            else:
                raise RuntimeError("Subprocess '" + secret + "' failed with exit code " + str(proc.returncode) )
      return outs.strip()
