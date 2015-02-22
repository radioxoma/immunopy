#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-11-16

@author: Eugene Dvoretsky
"""

__description__ = "Wrapper around Immunopy cli interface"
__author__ = "Eugene Dvoretsky"


import os
import csv
import itertools
import argparse
import subprocess
from functools import partial
from multiprocessing.dummy import Pool


def main(args):
    print(args)
    pool = Pool()
    protein_name = os.path.splitext(os.path.basename(args.file))[0]
    with open(args.settings, 'rb') as f:
        reader = csv.reader(f, delimiter=';')
        header = reader.next()
        col_idx = dict(itertools.izip(header, xrange(len(header))))
        # Now we can get a column index by name: `col_idx['Age']`
        settings_list = [row for row in reader]
    
    commands = list()
    for row in settings_list:
        dab_shift = int(row[col_idx['DAB shift']])
        hem_shift = int(row[col_idx['HEM shift']])
        fileout = os.path.join(args.out, protein_name + "_d%d-h%d.csv" % (dab_shift, hem_shift))
        shstr = "python2 cli.py %s %f --dab-shift %d --hem-shift %d --mp-disable --quiet --out %s" % (
            args.file, args.scale, dab_shift, hem_shift, fileout)
        commands.append(shstr)
    print(commands)
#     quit()
    for i, returncode in enumerate(pool.imap_unordered(partial(subprocess.call, shell=True), commands)):
        print("Let's play! %d" % i)
        if returncode != 0:
            print("%d command failed: %d" % (i, returncode))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument("file", help="An properly formatted file to process. E.g. 'ENSG00000148773.csv' (must be called same as 'protein_name' folder)")
    parser.add_argument("settings", help="Settings CSV list")
    parser.add_argument("scale", type=float, help="Image scale in um/px")
    parser.add_argument("out", help="Directory to save output files")
    main(parser.parse_args())

# "/home/radioxoma/analysis/hpa/ENSG00000148773.csv" "/home/radioxoma/analysis/hpa/settings_list.csv" 0.5 "/home/radioxoma/analysis/hpa/out/"
