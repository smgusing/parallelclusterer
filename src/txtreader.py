#! /usr/bin/env python
# -*- coding: utf-8 -*-
## Written by Gurpreet Singh##############

import  re
import numpy as np
import logging
logger = logging.getLogger(__name__)


def readcols(infile, skipline=1, readupto=0, dtype='float64'):
    'read file with data arranged in columns'
    logger.debug("File:-> %s", infile)
    try: 
        INPF = open(infile, 'r')
    except IOError:
        print "file", infile, " does not exist"
        raise SystemExit("Now Exiting")
    cols = []
    lineno, lines_read = 0, 0
    for line in INPF:
        if re.match("@|#|&", line):
            continue
        if (lineno % skipline == 0):
            line = line.split()
            # line=map(float,line.split())
            if len(line) < 1:
                logger.warn('empty line detected ... skipping')
                continue 
            lines_read += 1
            if ((readupto > 0) and (readupto == lines_read)): break
            cols.append(line)
        lineno += 1
    cols = np.array(cols, dtype=dtype)
    logger.info('File %s Lines read: %d',infile, lines_read)
    return cols
  
def readformat(infile, cols_type, read_nline=1, readupto=0):
    """ read files with format specified in cols_type and return numpy array
    """
    print "file is", infile
    try: 
        INPF = open(infile, 'r')
    except IOError:
        print "file", infile, " does not exist"
        exit(2)
    tcols = []
    lineno, lines_read = 0, 0
    for line in INPF:
        if re.match("@|#|&", line):
            continue
        # line=map(float,line.split())
        if (lineno % read_nline == 0):
            line = line.split()
            if len(line) < 1:
                print 'empty line detected ... skipping'
                continue 
            lines_read += 1
            if ((readupto > 0) and (readupto == lines_read)): break
            tcols.append(line)
        lineno += 1
    cols = np.zeros(len(tcols), cols_type)
    i = 0
    for entr in tcols:
        if (len(entr) > 1):
            entr = tuple(entr)
        else: entr = entr[0]
        cols[i] = entr
        i = i + 1
    cols = cols
    print 'Lines read: ', lines_read
    return cols
  
