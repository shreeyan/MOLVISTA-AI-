#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MolVista AI Plot Exporter (Plotly)

Generates PNG images from integrated prediction outputs:
- rmsd_hist.png
- centroid_hist.png
- rmsd_pie.png
- affinity_means_bar.png
- affinity_hist_<method>.png

Usage:
  python EquiBind/analysis/plot_results_plotly.py \
    --input EquiBind/results_demo/integrated_predictions.json \
    --out EquiBind/plots_demo
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go


def load_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [dict({'complex_name': k}, **v) for k, v in data.items()]
    return data


def extract_metrics(results):
    rmsd = []
    centroid = []
    affinities = defaultdict(list)
    for r in results:
        sm = r.get('structure_metrics', {})
        if sm.get('rmsd') is not None:
            rmsd.append(float(sm['rmsd']))
        if sm.get('centroid_distance') is not None:
            centroid.append(float(sm['centroid_distance']))
        for m, s in (r.get('affinity_scores') or {}).items():
            affinities[m].append(float(s))
    return np.array(rmsd), np.array(centroid), affinities


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save(fig, out_path, width=900, height=600):
    fig.write_image(out_path, scale=2, width=width, height=height)
    return out_path


def main():
    ap = argparse.ArgumentParser(description='MolVista AI: Export plots as PNG using Plotly')
    ap.add_argument('--input', required=True, help='Path to integrated_predictions.json')
    ap.add_argument('--out', required=True, help='Output directory for images')
    args = ap.parse_args()

    ensure_dir(args.out)
    results = load_results(args.input)
    rmsd, centroid, affinities = extract_metrics(results)

    generated = {}

    # RMSD histogram
    fig_rmsd = go.Figure(go.Histogram(x=rmsd, nbinsx=20, marker_color='#1f77b4'))
    fig_rmsd.update_layout(title='Docked Pose RMSD Distribution', xaxis_title='RMSD (Å)', yaxis_title='Count')
    generated['rmsd_hist'] = save(fig_rmsd, os.path.join(args.out, 'rmsd_hist.png'))

    # Centroid distance histogram
    fig_centroid = go.Figure(go.Histogram(x=centroid, nbinsx=20, marker_color='#ff7f0e'))
    fig_centroid.update_layout(title='Centroid Distance Distribution', xaxis_title='Centroid Distance (Å)', yaxis_title='Count')
    generated['centroid_hist'] = save(fig_centroid, os.path.join(args.out, 'centroid_hist.png'))

    # RMSD pie
    lt2 = int(np.sum(rmsd < 2.0))
    bt2_5 = int(np.sum((rmsd >= 2.0) & (rmsd < 5.0)))
    ge5 = int(np.sum(rmsd >= 5.0))
    fig_pie = go.Figure(go.Pie(labels=['<2 Å', '2–5 Å', '≥5 Å'], values=[lt2, bt2_5, ge5]))
    fig_pie.update_layout(title='Pose Quality by RMSD Thresholds')
    generated['rmsd_pie'] = save(fig_pie, os.path.join(args.out, 'rmsd_pie.png'))

    # Affinity means bar with error bars
    methods = []
    means = []
    stds = []
    for m, vals in affinities.items():
        if not vals:
            continue
        methods.append(m.capitalize())
        arr = np.array(vals)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr)))
    if methods:
        fig_bar = go.Figure(go.Bar(x=methods, y=means, error_y=dict(type='data', array=stds)))
        fig_bar.update_layout(title='Mean Affinity by Method', xaxis_title='Method', yaxis_title='Affinity')
        generated['affinity_means_bar'] = save(fig_bar, os.path.join(args.out, 'affinity_means_bar.png'))

    # Per-method affinity histograms
    generated['affinity_hists'] = []
    for m, vals in affinities.items():
        fig_h = go.Figure(go.Histogram(x=np.array(vals), nbinsx=20))
        fig_h.update_layout(title=f'Affinity Score Distribution ({m})', xaxis_title=f'{m.capitalize()} score', yaxis_title='Count')
        path = os.path.join(args.out, f'affinity_hist_{m}.png')
        save(fig_h, path)
        generated['affinity_hists'].append(path)

    with open(os.path.join(args.out, 'plots_index.json'), 'w') as f:
        json.dump(generated, f, indent=2)

    print('Generated images:')
    for k, v in generated.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()