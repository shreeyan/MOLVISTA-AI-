#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MolVista AI Result Plotting

Reads the integrated prediction outputs and generates publication-ready graphs:
- Histograms: RMSD, centroid distance, affinity per method
- Pie chart: pose RMSD categories (<2Å, 2–5Å, >=5Å)
- Bar chart: mean affinity per method with error bars
- Optional scatter: method vs method comparisons if multiple methods present

Usage:
    python EquiBind/analysis/plot_results.py \
        --input EquiBind/results_demo/integrated_predictions.json \
        --summary EquiBind/results_demo/prediction_summary.json \
        --out EquiBind/plots_demo

Notes:
- Expects `integrated_predictions.json` to be a list of result dicts containing:
  - `structure_metrics`: {`rmsd`, `centroid_distance`}
  - `affinity_scores`: {`neural`, `vina`, `physics`, `ensemble`} (subset)
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_results(input_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    # Expect a list of dicts
    if isinstance(data, dict):
        # Convert mapping to list for robustness
        data = [dict({'complex_name': k}, **v) for k, v in data.items()]
    return data


def extract_metrics_and_affinities(results):
    rmsd = []
    centroid = []
    affinities = defaultdict(list)
    for r in results:
        sm = r.get('structure_metrics', {})
        if sm.get('rmsd') is not None:
            rmsd.append(float(sm.get('rmsd')))
        if sm.get('centroid_distance') is not None:
            centroid.append(float(sm.get('centroid_distance')))
        for method, score in (r.get('affinity_scores') or {}).items():
            affinities[method].append(float(score))
    return np.array(rmsd), np.array(centroid), affinities


def ensure_out_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def plot_rmsd_hist(rmsd, out_dir):
    plt.figure(figsize=(7, 5))
    plt.hist(rmsd, bins=20, color='#1f77b4', alpha=0.85)
    plt.xlabel('RMSD (Å)')
    plt.ylabel('Count')
    plt.title('Docked Pose RMSD Distribution')
    plt.tight_layout()
    path = os.path.join(out_dir, 'rmsd_histogram.png')
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_centroid_hist(centroid, out_dir):
    plt.figure(figsize=(7, 5))
    plt.hist(centroid, bins=20, color='#ff7f0e', alpha=0.85)
    plt.xlabel('Centroid Distance (Å)')
    plt.ylabel('Count')
    plt.title('Centroid Distance Distribution')
    plt.tight_layout()
    path = os.path.join(out_dir, 'centroid_histogram.png')
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_rmsd_pie(rmsd, out_dir):
    values = rmsd
    cats = {
        '<2 Å': np.sum(values < 2.0),
        '2–5 Å': np.sum((values >= 2.0) & (values < 5.0)),
        '≥5 Å': np.sum(values >= 5.0)
    }
    labels = list(cats.keys())
    sizes = list(cats.values())
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Pose Quality by RMSD Thresholds')
    plt.tight_layout()
    path = os.path.join(out_dir, 'rmsd_piechart.png')
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_affinity_hists(affinities, out_dir):
    paths = []
    for method, scores in affinities.items():
        plt.figure(figsize=(7, 5))
        plt.hist(np.array(scores), bins=20, alpha=0.85)
        plt.xlabel('{} Score'.format(method.capitalize()))
        plt.ylabel('Count')
        plt.title('Affinity Score Distribution ({})'.format(method))
        plt.tight_layout()
        path = os.path.join(out_dir, 'affinity_hist_{}.png'.format(method))
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)
    return paths


def plot_affinity_bar_means(affinities, out_dir):
    methods = []
    means = []
    stds = []
    for method, scores in affinities.items():
        if not scores:
            continue
        methods.append(method)
        arr = np.array(scores)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr)))
    if not methods:
        return None
    x = np.arange(len(methods))
    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=4, color=['#66c2a5','#fc8d62','#8da0cb','#e78ac3'][:len(methods)], alpha=0.85)
    plt.xticks(x, [m.capitalize() for m in methods])
    plt.xlabel('Method')
    plt.ylabel('Mean Affinity (±1 SD)')
    plt.title('Mean Affinity by Method')
    plt.tight_layout()
    path = os.path.join(out_dir, 'affinity_means_bar.png')
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def plot_method_scatter(affinities, out_dir, m1='neural', m2='vina'):
    if m1 not in affinities or m2 not in affinities:
        return None
    a1 = np.array(affinities[m1])
    a2 = np.array(affinities[m2])
    n = min(len(a1), len(a2))
    if n < 2:
        return None
    plt.figure(figsize=(6, 6))
    plt.scatter(a1[:n], a2[:n], alpha=0.8)
    plt.xlabel('{}'.format(m1.capitalize()))
    plt.ylabel('{}'.format(m2.capitalize()))
    plt.title('{} vs {} Affinities'.format(m1.capitalize(), m2.capitalize()))
    plt.tight_layout()
    path = os.path.join(out_dir, 'scatter_{}_vs_{}.png'.format(m1, m2))
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description='MolVista AI: Plot results into graphs')
    parser.add_argument('--input', required=True, help='Path to integrated_predictions.json')
    parser.add_argument('--summary', required=False, help='Path to prediction_summary.json')
    parser.add_argument('--out', required=True, help='Directory to save plots')
    args = parser.parse_args()

    ensure_out_dir(args.out)
    results = load_results(args.input)
    rmsd, centroid, affinities = extract_metrics_and_affinities(results)

    generated = {}
    generated['rmsd_histogram'] = plot_rmsd_hist(rmsd, args.out)
    generated['centroid_histogram'] = plot_centroid_hist(centroid, args.out)
    generated['rmsd_piechart'] = plot_rmsd_pie(rmsd, args.out)
    generated['affinity_hists'] = plot_affinity_hists(affinities, args.out)
    generated['affinity_means_bar'] = plot_affinity_bar_means(affinities, args.out)
    generated['scatter_neural_vina'] = plot_method_scatter(affinities, args.out, 'neural', 'vina')

    # Save an index file listing generated plots
    index_path = os.path.join(args.out, 'plots_index.json')
    with open(index_path, 'w') as f:
        json.dump(generated, f, indent=2)

    print('Generated plots:')
    for k, v in generated.items():
        print('  {}: {}'.format(k, v))


if __name__ == '__main__':
    main()