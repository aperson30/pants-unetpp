# CPU-only small-lesion screen

The first 24 lexicographically sorted label files at the pinned MSD mirror
revision were downloaded and inspected sequentially. The remote log ended
with COMPLETE and inventory.json contained 24 records. No CTs or GPU jobs
were requested for this screen.

Smallest three positive total GT tumor burdens:

| Case | Tumor voxels | Total burden |
|---|---:|---:|
| pancreas_029 | 1167 | 1780.7007 mm3 |
| pancreas_028 | 1248 | 1981.2305 mm3 |
| pancreas_041 | 2112 | 2021.5837 mm3 |

None met the exploratory <=1000mm3 cutoff. This is total per-case tumor
burden, not connected-component volume, clinical stage, or a representative
dataset-wide distribution. The smaller cases have not been reconstructed.

The full inventory remains remote at
/projects/bdyo/asanjeev/compression_probe_20261001/small_inventory_24/inventory.json.
Its result was read over SSH; local artifact transfer failed after the SSH
master disappeared. A proposed next 48-mask CPU window was NOT launched.
The original remote inventory script is retained; the local updated script
adds window bounds and affine validation for the next invocation.

Additional GPU spending this screen: zero. Previous total remains 0.08528
physical GPU-hours / approximately 0.17056 charged allocation-equivalent hours.
Further comparable GPU reconstruction requires an additional approved cap.

## Connection restored; completed follow-up windows

The next two nonoverlapping 48-mask windows completed, for 120 unique masks
in total. All three complete inventories were retrieved into this results
directory. No GPU work was performed.

Smallest total positive GT tumor burden among these 120: pancreas_120,
454 voxels / 848.6181mm3 (0.849mL), native shape 512x512x104. This is the only
case in this screened subset meeting the exploratory <=1mL cutoff. It is
about nine times smaller in total burden than the initial pancreas_005 pilot.
Neither clinical stage nor representativeness follows from this selection.

`prepare_small_followup.py` selects the minimum from the complete inventories
before reconstruction, verifies pinned input/model hashes and geometry, and
downloads only the one matched CT. It does not submit any GPU job. Completion
of the download/geometry check must be verified separately before launch;
additional GPU-budget authorization is still required.

Preparation subsequently completed: matched case120 CT downloaded, image/mask
grid and millimeter units verified, affine axes checked orthonormal, input and
model hashes verified. The unchanged runner's CPU strict-checkpoint and 8-cubed
encode/decode preflight printed CPU_STRICT_CHECKPOINT_LOAD_OK. Manifest saved
as small_case_inputs_manifest.json. No reconstruction submitted or measured.
