#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-11-02

@author: Eugene Dvoretsky
"""

__description__ = "Images batch processing with Immunopy"
__author__ = "Eugene Dvoretsky"


import os
import argparse
import csv
import itertools
from scipy import misc 
import iptools
import lut


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
        # 'Antibody', 'Filename', 'Patient ID', 'Age', 'Staining', 'Intensity', 'Location', 'Quantity'
        assay_list = [row for row in reader]

    CMicro = iptools.CalibMicro(scale=args.scale)  # px/um
    CProcessor = iptools.CellProcessor(
        scale=CMicro.scale, colormap=lut.random_jet(), mp=True)    

    total_num = float(len(assay_list))
    result_list = list()
    for num, row in enumerate(assay_list, 1):
        ab_name = row[col_idx['Antibody']]
        image_name = row[col_idx['Filename']]
        img_path = os.path.join(csv_dir, protein_name, ab_name, image_name)
        print("\n[{}/{:2.0f} - {:3.0f} %] '{}/{}'").format(num, total_num, num / total_num * 100, ab_name, image_name)
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
        print(result_list[-1])
#         if num > 1:
#             break

    out_csv_filename = os.path.join(csv_dir, protein_name + '_ip.csv')
    with open(out_csv_filename, mode='wb') as f:
        writer = csv.writer(f, dialect=csv.excel, delimiter=';')
        header.extend(["DAB cell count", "HEM cell count", "DAB / DAB|HEM, %", "Compliance"])
        writer.writerow(header)
        for num, result in enumerate(result_list):
            assay_list[num].extend(result)
        writer.writerows(assay_list)
        print("\nFiles were written to '%s'") % out_csv_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument("file", help="An properly formatted file to process. E.g. 'ENSG00000148773.csv' (must be called same as protein_name folder)")
    parser.add_argument("scale", type=float, help="Image scale in ?")
    main(parser.parse_args())
