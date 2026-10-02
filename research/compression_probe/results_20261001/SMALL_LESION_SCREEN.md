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
