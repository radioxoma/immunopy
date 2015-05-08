#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-11-02

@author: Eugene Dvoretsky
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__description__ = "Human protein atlas images batch processing with Immunopy"
__author__ = "Eugene Dvoretsky"


import os
import argparse
import csv
import itertools
from scipy import misc

from . import iptools
from . import lut


def main(args):
    """Command line options.
    """
    csv_dir = os.path.dirname(args.file)
    protein_name = os.path.splitext(os.path.basename(args.file))[0]

    with open(args.file, 'rb') as f:
        reader = csv.reader(f, delimiter=';')
        header = reader.next()
        col_idx = dict(itertools.izip(header, xrange(len(header))))
        # Now we can get a column index by name: `col_idx['Age']`
        assay_list = [row for row in reader]

    CMicro = iptools.CalibMicro(scale=args.scale)  # px/um
    CProcessor = iptools.CellProcessor(
        scale=CMicro.scale, colormap=lut.random_jet(), mp=args.mp)
    if args.dab_shift is not None:
        CProcessor.th_dab_shift = args.dab_shift
    if args.hem_shift is not None:
        CProcessor.th_hem_shift = args.hem_shift
    total_num = float(len(assay_list))
    result_list = list()
    for num, row in enumerate(assay_list, 1):
        ab_name = row[col_idx['Antibody']]
        image_name = row[col_idx['Filename']]
        img_path = os.path.join(csv_dir, protein_name, ab_name, image_name)
        print("[{}/{:2.0f} - {:3.0f} %] '{}/{}'").format(num, total_num, num / total_num * 100, ab_name, image_name)
        if not args.dry_run:
            rgb = misc.imread(img_path)
            CProcessor.process(rgb)
        CProcessor.st_dabdabhem_fraction

        if row[col_idx['Quantity']] == 'gt75%':
            compliance = 75 < CProcessor.st_dabdabhem_fraction
        elif row[col_idx['Quantity']] == '75%-25%':
            compliance = 25 < CProcessor.st_dabdabhem_fraction < 75
        elif row[col_idx['Quantity']] == 'lt25%':
            compliance = CProcessor.st_dabdabhem_fraction < 25
        elif row[col_idx['Quantity']] == 'Rare':
            compliance = CProcessor.st_dabdabhem_fraction < 14
        elif row[col_idx['Quantity']] == 'Negative':
            compliance = CProcessor.st_dabdabhem_fraction < 3
        else:
            raise ValueError("Wrong table value")

        result_list.append(
            [CProcessor.st_dab_cell_count,
             CProcessor.st_hem_cell_count,
             CProcessor.st_dabdabhem_fraction,
             int(compliance)])
        if not args.quiet:
            print(result_list[-1])
#         if num > 1:  # Break run for testing
#             break

    if args.out:
        out_csv_filename = os.path.join(os.getcwdu(), args.out)
    else:
        out_csv_filename = os.path.join(csv_dir, protein_name + '_ip.csv')

    if args.dry_run:
        print("Will be writing to %s") % out_csv_filename
    else:
        with open(out_csv_filename, mode='wb') as f:
            writer = csv.writer(f, dialect=csv.excel, delimiter=';')
            header.extend(["DAB cell count", "HEM cell count", "DAB / DAB|HEM, %", "Compliance"])
            writer.writerow(header)
            for num, result in enumerate(result_list):
                assay_list[num].extend(result)
            writer.writerows(assay_list)
            if not args.quiet:
                print("\nFiles were written to '%s'") % out_csv_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument("file", help="An properly formatted file to process. E.g. 'ENSG00000148773.csv' (must be called same as 'protein_name' folder)")
    parser.add_argument("scale", type=float, help="Image scale in um/px")
    parser.add_argument("--out", help="Output filename (it's a CSV file). If not provided, image will be saved in the same directory as input file")
    parser.add_argument("--dab-shift", type=int, help="DAB threshold shift")
    parser.add_argument("--hem-shift", type=int, help="HEM threshold shift")
    parser.add_argument("--mp-disable", dest='mp', action='store_false', help="Disable multiprocessing")
    parser.add_argument("--quiet", action='store_true', help="Do not say anything")
    parser.add_argument("--dry-run", action='store_true', help="Do not write anything")
    main(parser.parse_args())
