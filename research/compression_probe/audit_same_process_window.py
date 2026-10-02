"""CPU saved-map verification; no new predictions or clinical threshold."""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from judge_contracts import crop_slices


def audit(producer,label):
    completion=json.loads((producer/'completion.json').read_text())
    if completion.get('completed') is not True or completion.get('historical_replay_gate_passed') is not False:
        raise ValueError('Require separate completed fresh-control study')
    tumor=sitk.GetArrayFromImage(sitk.ReadImage(str(label)))==1
    rows={}
    for row in completion['results']:
        image_arm,crop_arm=row['image_arm'],row['crop_arm']
        key=(image_arm,crop_arm)
        if key in rows or image_arm not in ('native','reconstruction') or crop_arm not in ('native','reconstruction'):
            raise ValueError('Unexpected/duplicate factorial cell')
        window=row['crop_bounds']
        local_tumor=tumor[crop_slices(tumor.shape,window)]
        # Means below reconstruct full-grid scoring, including any excluded GT.
        # Never silently drop out-of-window tumor voxels from the denominator.
        with np.load(producer/(image_arm+'_crop_'+crop_arm)/'crop_maps.npz',allow_pickle=False) as maps:
            raw,candidates,mask=(maps[k] for k in ('raw_pdac','candidates','postprocess_mask'))
            if any(x.shape!=local_tumor.shape or not np.isfinite(x).all() for x in (raw,candidates,mask)):
                raise ValueError('Saved map geometry/nonfinite mismatch')
            checks=dict(tumor_mean_raw_probability=float(raw[local_tumor].sum()/tumor.sum()),
                tumor_max_raw_probability=float(raw[local_tumor].max()) if local_tumor.any() else 0.,
                candidate_confidence_on_tumor=float(candidates[local_tumor].max()) if local_tumor.any() else 0.,
                tumor_voxels_with_candidate=int(np.count_nonzero(candidates[local_tumor])))
        for name,value in checks.items():
            if not np.isclose(value,row[name],rtol=1e-6,atol=1e-8):
                raise ValueError('Saved map/producer scalar mismatch: '+name)
        repeat=row['repeatability']
        if repeat['all_class_max_abs_probability_drift']>1e-4 or repeat['segmentation_labels_different']!=0:
            raise ValueError('Failed repeated-control cell')
        rows[key]=row
    if len(rows)!=4:
        raise ValueError('Incomplete factorial output')
    contrasts=[]
    for crop_arm in ('native','reconstruction'):
        original=rows['native',crop_arm]['tumor_mean_raw_probability']
        reconstructed=rows['reconstruction',crop_arm]['tumor_mean_raw_probability']
        contrasts.append(dict(crop_arm=crop_arm,native_tumor_mean=original,
            reconstruction_tumor_mean=reconstructed,
            reconstruction_native_ratio=reconstructed/original if original else None,
            reconstruction_native_signed_change=reconstructed-original))
    return dict(completed=True,clinical_recall=False,historical_gate_retroactively_passed=False,
        saved_map_scalar_parity=True,contrasts=contrasts,
        window_changes=[dict(image_arm=arm,tumor_mean_change_reconstruction_window_minus_native_window=
            rows[arm,'reconstruction']['tumor_mean_raw_probability']-rows[arm,'native']['tumor_mean_raw_probability'])
            for arm in ('native','reconstruction')],
        caveat='One posthoc patient; fixed windows do not exclude domain shift or prove clinical loss/information erasure')


def main():
    parser=argparse.ArgumentParser()
    for name in ('producer','label','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.producer,args.label)
    with args.output.open('x') as output:
        json.dump(result,output,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':
    main()
