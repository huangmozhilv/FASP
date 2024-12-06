import os
import sys
import itertools
import time
import shutil
import pdb

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    e.g. 
    from ccToolkits import tinies
    tinies.ForkedPdb().set_trace()
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def newDir(path):
    # always make new dir
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def sureDir(path):
    # only make new dir when not existing
    os.makedirs(path, exist_ok=True)
    # os.makedirs(path)

def ForceCopyDir(oldPath, newPath, ignore=None):
    '''
    copy dir from oldPath to dir in newPath. Overwritten.
    '''
    if os.path.exists(newPath):
        shutil.rmtree(newPath)
    else:
        pass
    shutil.copytree(oldPath, newPath, ignore=ignore) # target dir should always be removed. here use shutil.rmtree() as above

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

def timer(start, end):
    '''
    end-start: returns seconds
    '''
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "h:m:s, {:0}:{:0>2}:{:0>2}".format(int(hours),int(minutes),int(seconds))
