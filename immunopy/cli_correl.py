#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__author__ = "Eugene Dvoretsky"

import argparse
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd



def main():
    parser = argparse.ArgumentParser(description="Calculate correlation.")
    parser.add_argument('file', help="An CSV file")
    args = parser.parse_args()

    csv_in = pd.read_csv(args.file)
    human_list = csv_in['Human labeling index']
    ip_list = csv_in['DAB / DAB|HEM, %']
    stat = "Pearson {0:.3f}, p={1:.3f};\n".format(*stats.pearsonr(human_list, ip_list))
    stat += "Spearman {0:.4f}, p={1:.3f}".format(*stats.spearmanr(human_list, ip_list))
    print(stat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(stat)

    plt.xlabel("Human observer")
    plt.ylabel("Immunopy")
    plt.ylim([0, 100])
    plt.xlim([0, 100])
    plt.plot([0, 100], [1, 100])
    ax.add_patch(patches.Rectangle((0,0), 25, 25, alpha=0.04))
    ax.add_patch(patches.Rectangle((0,0), 3, 3, alpha=0.05, label="Negative"))
    ax.add_patch(patches.Rectangle((3,3), 15-3, 15-3, alpha=0.09, label="Rare"))
    ax.add_patch(patches.Rectangle((15,15), 25-15, 25-15, alpha=0.13, label="<25%"))
    ax.add_patch(patches.Rectangle((25,25), 75-25, 75-25, alpha=0.17, label="75%-25%"))
    ax.add_patch(patches.Rectangle((75,75), 100-75, 100-75, alpha=0.23, label=">75%"))

    xy = np.vstack([human_list, ip_list])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = human_list[idx], ip_list[idx], z[idx]
    ax.add_collection(plt.scatter(x, y, s = 200, c = z, edgecolor=''))#, cmap='OrRd'))

    # ax.legend(loc="upper left")`
    ax.set_aspect('equal', 'datalim')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
