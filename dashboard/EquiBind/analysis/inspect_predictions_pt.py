#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys

def main():
    try:
        import torch
    except Exception as e:
        print('Torch not available:', e)
        return

    pt_path = os.path.join('EquiBind','runs','flexible_self_docking','predictions_RDKitFalse.pt')
    print('PT exists:', os.path.exists(pt_path))
    if not os.path.exists(pt_path):
        return

    try:
        obj = torch.load(pt_path, map_location='cpu')
        print('Loaded type:', type(obj))
        if isinstance(obj, dict):
            print('Keys:', list(obj.keys()))
            # Prefer corrected_predictions if available
            preds = obj.get('corrected_predictions') or obj.get('predictions') or obj.get('initial_predictions')
            targs = obj.get('targets')
            names = obj.get('names')
            if preds is not None and targs is not None:
                try:
                    import numpy as np
                    print('Num preds:', len(preds), 'Num targets:', len(targs))
                    rmsds = []
                    for i in range(len(preds)):
                        p = preds[i]
                        t = targs[i]
                        p_np = p.numpy() if hasattr(p, 'numpy') else np.array(p)
                        t_np = t.numpy() if hasattr(t, 'numpy') else np.array(t)
                        rmsd = np.sqrt(np.mean(np.sum((p_np - t_np)**2, axis=1)))
                        rmsds.append(float(rmsd))
                    n = len(rmsds)
                    succ2 = sum(1 for x in rmsds if x <= 2.0)
                    succ5 = sum(1 for x in rmsds if x <= 5.0)
                    rate2 = (float(succ2)/float(n)*100.0) if n else 0.0
                    rate5 = (float(succ5)/float(n)*100.0) if n else 0.0
                    print('Computed from PT:')
                    print('Count:', n)
                    print('Success@2A: {:.1f}%'.format(rate2))
                    print('Success@5A: {:.1f}%'.format(rate5))
                except Exception as e:
                    print('Error computing RMSD from PT:', e)
        else:
            print('Non-dict PT content.')
    except Exception as e:
        print('Load error:', e)

if __name__ == '__main__':
    main()