# ABIDE atlas identity and exclusion audit

A post-pilot source audit finds a limitation of the frozen 111-dimensional
brain study: its mask contains 110 named regions and unnamed label 3455,
occupying two 3-mm voxels (54 mm³). That label is constant in 172 of 780 eligible
scans and is the sole constant series in 161 of 179 rejected scans. These are
source diagnostics, not new model results or a change to the original data.

## Pinned source evidence

The PCP [ROI list](https://github.com/preprocessed-connectomes-project/abide/blob/fda1ec5d3f53eb97d94352e779f37c3ce46caee1/preprocessing/settings/abide_rois.txt)
points to the [historical mask](https://github.com/preprocessed-connectomes-project/abide/blob/fda1ec5d3f53eb97d94352e779f37c3ce46caee1/preprocessing/resources/abide_rois/ho_mask_pad.nii.gz).
Its SHA-256 is
`e2ef01f7394a5d0dde7c8cd57f8cc72079b2115b4180d6466aafb0831e08f8c1`.
The pinned [C-PAC template pointer](https://github.com/FCP-INDI/C-PAC_templates/blob/b64c29d83a015e92966ca0371370b0f2ceda7365/atlases/label/Human/ho_mask_pad.nii.gz)
identifies exactly the same 12,567-byte object.

PCP's [pipeline documentation](https://github.com/preprocessed-connectomes-project/abide/blob/21dae10aaf2b6d576e9f9eee29e48ea6a875daed/Pipelines.md)
describes a 25% threshold, hemisphere splitting, removal of selected subcortical
non-gray-matter regions and resampling to functional resolution. It links the
[hosted atlas](https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/ho_roi_atlas.nii.gz)
and [label table](https://fcp-indi.s3.amazonaws.com/data/Projects/ABIDE_Initiative/Resources/ho_labels.csv).
The hosted atlas has a different compressed hash but exactly the same voxel
array and affine as the historical mask. Both contain all 111 pilot IDs. The
CSV names 110 distinct IDs, all present in the mask; only 3455 lacks a name.
We do not infer an anatomical name or intent for that label.

The [machine-readable audit](abide_atlas_audit_20261006.json) records resource,
input-manifest and audit-code hashes, shape, resolution and per-site counts.
It verifies all 780 input files and the exact membership of all 179 exclusions.
This is not a reproduction of the upstream raw-image preprocessing.

## Effect on the frozen cohort

| Partition | Frozen retained | Excluded solely for 3455 | Has another constant region | Nonconstant after removing 3455 |
| --- | ---: | ---: | ---: | ---: |
| Train | 349 | 139 | 16 | 488 |
| Michigan validation | 92 | 16 | 0 | 108 |
| NYU test | 160 | 6 | 2 | 166 |
| Total | 601 | 161 | 18 | 762 |

Eleven of the 18 subjects with other constant regions also have constant label
3455. The table describes eligibility, not a newly built 762-subject dataset or
evidence that model performance improves. The original and five-training-seed
results still describe the frozen 601-subject cohort and establish no augmentation
gain there. They do not validate this atlas choice.

Prespecification alone is insufficient: a mechanically applied rule can select
subjects for reasons unrelated to the intended anatomical target. The audit
retains site-specific effects. Matrix positive definiteness cannot reveal this
observation-level issue.

## Separate follow-up

Preserve frozen arrays, checkpoints, metrics and leaderboard records. Remove
3455 from regional series before standardization and OAS estimation; deleting a
cached matrix row/column does not reproduce re-estimation of shrinkage. Declare:

1. A comparison on the same 601 subjects, isolating the target change as far as
   possible.
2. A comparison on all 762 nonconstant subjects, explicitly changing both target
   and cohort.

Rerun real-only/replay/Gaussian controls before learned fits and preserve negative
outcomes. Use a separate training/selection protocol. Already inspected sites
remain inspected. At the time of this source audit, the follow-up was unbuilt.
The later [declared 110-region control study](abide_named110_protocol_20261006.md)
now has [complete results](abide_named110_results_20261006.md): both reconstructions
and real/replay/Gaussian controls, with no learned 110-region fit.

## Reproduce

Download the three linked atlas resources into a directory using their displayed
filenames; save the pinned C-PAC pointer's raw text as `cpac_mask_pointer`.
The script requires NumPy and NiBabel (tested 2.5.3 and 5.4.2). Existing pilot
source files remain local under their original terms.

```bash
python research/audit_abide_atlas.py \
  --resources artifacts/abide_provenance_20261006 \
  --pilot artifacts/abide_ho_20261005 \
  --series artifacts/abide_source_20261005/rois_ho \
  --output artifacts/abide_atlas_new_audit.json
```

The output must not exist. The script verifies resource hashes, atlas arrays and
affines, label IDs, all series hashes and frozen exclusion membership. It uses
no GPU or model training.

Historical mask identity is now established. License scope and the full
ABIDE/PCP/atlas/site attribution record remain separate; see the
[release review](abide_release_review.md). No atlas image or participant matrix
was uploaded. A revised 110-label dataset needs a separate version and manifest.
