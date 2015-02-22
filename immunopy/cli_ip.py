#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

"""
Created on 2015-02-14

@author: Eugene Dvoretsky

python2 cli_ip.py ~/analysis/Лесничая\ О.В./Immunopy/2015-01-26/2015-01-26_rescaled_analyzed/ ~/analysis/Лесничая\ О.В./Immunopy/2015-01-26/origin/flat/ 0.5 -o here.csv --dab-shift -30 --hem-shift -20
"""

__description__ = "Image batch processing utility."
__author__ = "Eugene Dvoretsky"

import argparse
import csv
import os
import re
import itertools
from scipy import misc
import iptools
import lut


def parse_name(filename):
    """
    Percent is not digit! Also it can be str('na').
    """
    # Search original TIFF name and human labeling index
    # example: `20150126-145740[35].tif`
    match = re.search(r"(^.+)\[(.+(?=\]))", filename)
    identifer = match.group(1)  # 20150126-145740
    percent = match.group(2)  # 35
    return identifer, percent


def main():
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument("rated", help="Directory with images 20150126-145740[35].tif.jpg")
    parser.add_argument("scale", type=float, help="Image scale in um/px")
    parser.add_argument("-o", "--out", help="Output CSV file")
    parser.add_argument("--dab-shift", type=int, help="DAB threshold shift")
    parser.add_argument("--hem-shift", type=int, help="HEM threshold shift")
    parser.add_argument("--mp-disable", dest='mp', action='store_false', help="Disable multiprocessing")
    parser.add_argument("--dry-run", action='store_true', help="Do not process anything")
    args = parser.parse_args()

    # header = ["Assay ID", "Human labeling index"]
    header = ["Assay ID", "Human labeling index", "DAB cell count", "HEM cell count", "DAB / DAB|HEM, %"]
    col_idx = dict(itertools.izip(header, xrange(len(header))))


    # Prepare image processing
    CMicro = iptools.CalibMicro(objective="20")
    CProcessor = iptools.CellProcessor(
        scale=CMicro.scale, colormap=lut.random_jet(), mp=args.mp)
    CProcessor.vtype = 'Overlay'
    if args.dab_shift is not None:
        CProcessor.th_dab_shift = args.dab_shift
    if args.hem_shift is not None:
        CProcessor.th_hem_shift = args.hem_shift
    print("DAB shift {}, HEM shift {}".format(CProcessor.th_dab_shift, CProcessor.th_hem_shift))

    assay_list = list()
    for filename in filter(lambda x: x.endswith('.tif') and '[' in x, sorted(os.listdir(args.rated))):
        # Search original TIFF name and human labeling index
        try:
            _, p = parse_name(filename)
            if p.isdigit():
                assay_list.append([filename, p])
        except:
            print(filename)
            raise

    total_num = len(assay_list)
    for num, row in enumerate(assay_list):
        image_name = row[col_idx['Assay ID']]
        print("[{}/{:2.0f} - {:3.0f} %] '{}'").format(num, total_num, num / total_num * 100, image_name)
        if not args.dry_run:
            CProcessor.process(misc.imread(os.path.join(args.rated, image_name)))
            assay_stat = CProcessor.take_assay()
        assay_list[num].extend([assay_stat.dab_cell_count, assay_stat.hem_cell_count, assay_stat.dab_dabhemfraction])

    if args.out:
        with open(args.out, 'wb') as f:
            writer = csv.writer(f, dialect=csv.excel)
            writer.writerow(header)
            writer.writerows(assay_list)
    else:
        for k in assay_list:
            print(k)


if __name__ == '__main__':
    main()
