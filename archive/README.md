# Archive — superseded scripts

Kept for provenance only. **Do not use these**; the working equivalents are in `evaluation/`.

| File | Superseded by | Why |
|---|---|---|
| `compute_pants_metrics.py` | `evaluation/compute_tumor_metrics.py` | Earlier metrics implementation, used for the first completed UNet++ run. Its per-case output (`per_case_results.csv` on the server) confirmed the newer script agrees with it. |
| `extract_max_probs.py` | `evaluation/predict_and_shrink.py` | **Contained a real bug:** deleted each original probability file *before* its extracted value was written to disk. Combined with an SSH disconnect, this permanently destroyed data for 346 test cases. The replacement writes first, deletes after. |
| `combine_metrics.py` | folded into `compute_tumor_metrics.py` | Merged per-part CSVs from parallel prediction runs. |
| `check_npz_integrity.py` | `evaluation/verify_and_repair_predictions.py` | Only checked files existed; the replacement loads each one to catch silent truncation. |
| `clean_corrupted_predictions.py` | `evaluation/verify_and_repair_predictions.py` | Manual cleanup; the replacement repairs and re-verifies automatically. |
| `run_overnight.sh` | `evaluation/post_train_scheduler.sh` | Fixed sequence; the replacement dispatches opportunistically as GPUs free up. |

The `extract_max_probs.py` incident is the origin of the standing rule to wrap every server command
in `nohup`, however trivial it looks.
