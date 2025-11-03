#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import os
import sys
import argparse


def compute_from_json(json_path):
    # Open without encoding for Python 2 compatibility
    with open(json_path, 'r') as f:
        data = json.load(f)

    rmsds = []
    centroids = []
    for item in data:
        if isinstance(item, dict):
            rmsd = None
            # common locations for RMSD
            if 'rmsd' in item:
                rmsd = item.get('rmsd')
            elif 'structure_metrics' in item:
                rmsd = item['structure_metrics'].get('rmsd')
                cd = item['structure_metrics'].get('centroid_distance')
                if cd is not None:
                    try:
                        centroids.append(float(cd))
                    except Exception:
                        pass
            elif 'metrics' in item:
                rmsd = item['metrics'].get('rmsd')
            if rmsd is not None:
                try:
                    rmsds.append(float(rmsd))
                except Exception:
                    pass

    n = len(rmsds)
    mean_rmsd = sum(rmsds) / n if n else float('nan')
    if n:
        s = sorted(rmsds)
        mid = n // 2
        if n % 2 == 0:
            median_rmsd = (s[mid - 1] + s[mid]) / 2.0
        else:
            median_rmsd = s[mid]
    else:
        median_rmsd = float('nan')

    succ_2 = sum(1 for x in rmsds if x <= 2.0)
    succ_5 = sum(1 for x in rmsds if x <= 5.0)
    rate_2 = (float(succ_2) / float(n) * 100.0) if n else 0.0
    rate_5 = (float(succ_5) / float(n) * 100.0) if n else 0.0

    # centroid stats
    cn = len(centroids)
    if cn:
        cs = sorted(centroids)
        cmid = cn // 2
        if cn % 2 == 0:
            median_centroid = (cs[cmid - 1] + cs[cmid]) / 2.0
        else:
            median_centroid = cs[cmid]
        mean_centroid = sum(centroids) / float(cn)
    else:
        mean_centroid = float('nan')
        median_centroid = float('nan')

    print("Count:", n)
    print("Mean RMSD (A):", round(mean_rmsd, 3))
    print("Median RMSD (A):", round(median_rmsd, 3))
    print("Success@2A: {} / {} ({:.1f}%)".format(succ_2, n, rate_2))
    print("Success@5A: {} / {} ({:.1f}%)".format(succ_5, n, rate_5))
    print("Mean Centroid Distance (A):", round(mean_centroid, 3))
    print("Median Centroid Distance (A):", round(median_centroid, 3))

    return {
        'count': n,
        'mean_rmsd': mean_rmsd,
        'median_rmsd': median_rmsd,
        'success_2A': succ_2,
        'success_5A': succ_5,
        'rate_2A': rate_2,
        'rate_5A': rate_5,
        'mean_centroid': mean_centroid,
        'median_centroid': median_centroid,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute docking accuracy from integrated_predictions.json")
    parser.add_argument("json_path", type=str, help="Path to integrated_predictions.json")
    parser.add_argument("--out_json", type=str, default=None, help="Optional path to write summary JSON")
    parser.add_argument("--out_txt", type=str, default=None, help="Optional path to write summary text")
    args = parser.parse_args()
    summary = compute_from_json(args.json_path)
    if args.out_json:
        try:
            with open(args.out_json, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass
    if args.out_txt:
        try:
            with open(args.out_txt, 'w') as f:
                f.write("Count: {}\n".format(summary['count']))
                f.write("Mean RMSD (A): {:.3f}\n".format(summary['mean_rmsd']))
                f.write("Median RMSD (A): {:.3f}\n".format(summary['median_rmsd']))
                f.write("Success@2A: {} / {} ({:.1f}%)\n".format(summary['success_2A'], summary['count'], summary['rate_2A']))
                f.write("Success@5A: {} / {} ({:.1f}%)\n".format(summary['success_5A'], summary['count'], summary['rate_5A']))
                f.write("Mean Centroid Distance (A): {:.3f}\n".format(summary['mean_centroid']))
                f.write("Median Centroid Distance (A): {:.3f}\n".format(summary['median_centroid']))
        except Exception:
            pass


if __name__ == "__main__":
    main()