"""CPU-only independent saved-map audit; all locked seeds, no clinical recall."""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from judge_contracts import crop_slices


def audit(producer,label):
    completion=json.loads((producer/'completion.json').read_text())
    if completion.get('completed') is not True or completion.get('seeds')!=[0,1,2]:
        raise ValueError('Require completed locked three-seed study')
    tumor=sitk.GetArrayFromImage(sitk.ReadImage(str(label)))==1
    if not tumor.any():raise ValueError('Empty tumor')
    rows={}
    for row in completion['results']:
        arm=row['arm']
        if arm in rows:raise ValueError('Duplicate arm')
        local_tumor=tumor[crop_slices(tumor.shape,row['crop_bounds'])]
        with np.load(producer/arm/'crop_maps.npz',allow_pickle=False) as maps:
            raw,masked,candidates,mask=(maps[k] for k in ('raw_pdac','masked_pdac','candidates','postprocess_mask'))
            if any(x.shape!=local_tumor.shape or not np.isfinite(x).all() for x in (raw,masked,candidates,mask)):
                raise ValueError('Invalid saved maps')
            checks=dict(tumor_mean_raw_probability=float(raw[local_tumor].sum()/tumor.sum()),
                tumor_max_raw_probability=float(raw[local_tumor].max()) if local_tumor.any() else 0.,
                tumor_mean_masked_probability=float(masked[local_tumor].sum()/tumor.sum()),
                candidate_confidence_on_tumor=float(candidates[local_tumor].max()) if local_tumor.any() else 0.,
                tumor_voxels_with_candidate=int(np.count_nonzero(candidates[local_tumor])),
                tumor_fraction_in_postprocess_mask=float(mask[local_tumor].sum()/tumor.sum()))
        for name,value in checks.items():
            if not np.isclose(value,row[name],rtol=1e-6,atol=1e-8):
                raise ValueError('Saved-map/producer mismatch: '+arm+'/'+name)
        if row['repeatability']['all_class_max_abs_probability_drift']>1e-4 or row['repeatability']['segmentation_labels_different']!=0:
            raise ValueError('Failed repeat gate')
        rows[arm]=row
    expected=('native','control','posterior_mean','posterior_seed0','posterior_seed1','posterior_seed2')
    if set(rows)!=set(expected) or any(rows[arm]['crop_bounds']!=rows['native']['crop_bounds'] for arm in expected):
        raise ValueError('Missing/extra arm or changed window')
    reference=rows['native']['tumor_mean_raw_probability']
    if reference<=0:raise ValueError('Native ratio undefined')
    contrasts=[dict(arm=arm,tumor_mean_raw_probability=rows[arm]['tumor_mean_raw_probability'],
        ratio_to_native=rows[arm]['tumor_mean_raw_probability']/reference,
        candidate_confidence_on_tumor=rows[arm]['candidate_confidence_on_tumor'],
        tumor_voxels_with_candidate=rows[arm]['tumor_voxels_with_candidate'],
        tumor_fraction_in_crop=rows[arm]['tumor_fraction_in_crop'],
        tumor_fraction_in_postprocess_mask=rows[arm]['tumor_fraction_in_postprocess_mask']) for arm in expected]
    return dict(completed=True,clinical_recall=False,saved_map_scalar_parity=True,
        all_seeds_reported=True,contrasts=contrasts,
        caveat='One feasibility-selected patient and one frozen judge; posterior control not diffusion generation, clinical loss or irreversible erasure')


def main():
    parser=argparse.ArgumentParser()
    for name in ('producer','label','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=audit(args.producer,args.label)
    with args.output.open('x') as output:json.dump(result,output,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':main()
