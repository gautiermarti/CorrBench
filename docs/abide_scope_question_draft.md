# Draft clarification on atlas-derived numerical matrices

Prepared for review; **not sent**. No contact or response is claimed. Email
integration remains deferred in the research project.

The specific question is for the Harvard–Oxford/FSL atlas licensing contact
listed on the [official license page](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html),
with ABIDE/PCP consulted if their derivative-distribution interpretation is also
needed. This is not a request for unrestricted or commercial ABIDE rights.

> We are preparing a noncommercial research reference dataset derived from the
> official ABIDE Preprocessed Connectomes Project's
> `CPAC/filt_noglobal/rois_ho` regional time series. The files would contain
> per-subject Pearson and OAS correlation matrices, integer ROI IDs, public
> source subject/site codes and the already published diagnosis, age, sex and
> motion fields. They would include no raw MRI, regional time series, atlas
> voxels/images or copied atlas-name table. Complete ABIDE, PCP, atlas,
> investigator and funder credits, source links and transformation notices
> accompany the files.
>
> ABIDE links CC BY-NC-SA 3.0; the current FSL page lists Harvard–Oxford under
> CC BY-SA 4.0. Does the latter license apply to these numerical correlation
> outputs as adaptations of the atlas, or does it apply to redistribution of
> the atlas itself? May these matrix-only ABIDE derivatives be distributed
> under the ABIDE CC BY-NC-SA 3.0 terms with atlas attribution? Please also
> identify any additional terms applicable to the historical PCP
> `ho_mask_pad.nii.gz` used by `rois_ho`.
>
> The candidate contains three explicitly separated views: the frozen original
> 111-ID/601-subject pilot, the same 601 subjects re-estimated with 110 named
> regions, and the expanded 762-subject named-region cohort. The original
> mask's unnamed two-voxel ID 3455 and its exclusion effect are documented.
> These views overlap and are not a protected competition holdout.

Supporting material: [scope review](abide_release_review.md),
[mask provenance](abide_atlas_audit_20261006.md),
[attribution and cohort counts](abide_attribution_20261006.md), and
[local reference audit](abide_reference_complete_20261006.json).

An answer should address the actual file contents and historical mask, rather
than only restating either license name. The software/NOAA/toy releases can
continue independently while this specific question remains open.
