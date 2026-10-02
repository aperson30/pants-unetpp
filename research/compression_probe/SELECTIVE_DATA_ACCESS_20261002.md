# Selective PANORAMA access: CPU-only feasibility result

The public version-1 Zenodo batch archive honored bounded HTTP Range requests:

- URL: https://zenodo.org/records/10998332/files/batch_1.zip?download=1
- Archive length: 48,238,512,666 bytes.
- ZIP directory: 554 members, obtained with 49,683 bytes across 5 requests.
- Example `batch_1/100028_00001_0000.nii.gz`: 49,164,104 bytes,
  ZIP_STORED (compression method 0), local-header offset 0.
- No member was opened, downloaded, extracted or decoded. No GPU allocation.

Evidence stays in `/u/asanjeev/compression_probe_smalloutputs/` on DeltaAI:
`panorama_batch1_directory_v1.json` and its companion log. This establishes
directory access, NOT a successfully extracted CT or an eligible held-out case.
The earlier latest-version request returned 429; we stopped that request.
Version-1 access must not be silently mixed with another release's metadata.

## Reproducible metadata-only tool

`list_remote_zip.py` uses the standard ZIP parser over a bounded range reader.
It refuses HTTP 200 responses before reading their body, checks Content-Range,
limits each request to 4 MiB, all fetched bytes to 8 MiB and calls to 32.
Each request has a 15-second timeout; the remote run also had a 90-second
process timeout. Unbounded reads are prohibited except the parser's small EOF
directory tail. Three local and remote unit tests passed. The final source
additionally reserves the oversize-detection byte within its total-byte bound;
all three local tests passed again after that tightening.

## Cheapest next gate, before another GPU job

1. Read the public clinical workbook and manual-label IDs; join exact study IDs
   to this archive directory and the pinned split/checkpoint identity evidence.
   The labels repo currently has `clinical_information.xlsx`, not a root CSV.
2. Select a small prespecified set eligible for the intended held-out comparison;
   exclude patient overlap and unresolved dataset-version or selection exposure.
   Do not cherry-pick cases based on reconstruction or detector outcomes.
3. Implement a bounded single-member retrieval only after eligibility is settled:
   verify the local header, exact filename, compression/encryption flags, sizes,
   stable source identity, ZIP CRC and complete NIfTI geometry/finite data. Abort
   on ignored Range, quota/write failure, inconsistent metadata or rate limits.
   CRC checks transfer integrity, not authenticity; preserve release provenance
   and a SHA256 of the fetched file as well.
4. Check the intended judge's task, input convention and held-out provenance.
   Frozen-detector failure alone is not proof of irrecoverable information loss.
5. Only then price a tiny paired real-lesion test under the remaining authorized
   two-hour total. Preserve original-image misses and paired false positives.

No checkpoint download, clinical-workbook analysis, individual CT extraction,
reader-key publication or GPU submission occurred in this access probe. This
does not touch PanTS evaluation job 47320183. No large download is justified
merely because directory access succeeded.
