# ABIDE source terms: evidence collected for release review

Updated 2026-10-06. The staged brain reference remains unpublished.

The [atlas audit](abide_atlas_audit_20261006.md) now resolves historical mask
identity: the pinned PCP mask matches the hosted atlas voxel-for-voxel and the
current C-PAC template points to the same bytes. It also identifies an unnamed
two-voxel label, 3455, responsible alone for 161 of 179 exclusions. Preserve the
111-dimensional data as the original pilot, with this limitation.

The [ABIDE I usage page](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html)
links specifically to
[CC BY-NC-SA 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0/).
It describes non-commercial research use, asks authors to identify included
datasets and acknowledge funding, and describes NITRC/INDI registration for its
download route. This identifies the license version that was previously open.
The project's [preprocessed-data page](https://fcon_1000.projects.nitrc.org/indi/abide/preprocessed.html)
links the PCP derivative collection used here.

The official [PCP download guide](https://github.com/preprocessed-connectomes-project/abide/blob/fda1ec5d3f53eb97d94352e779f37c3ce46caee1/download_abide_preproc_guide.txt)
documents its public S3 route, CPAC, the filt_noglobal strategy and rois_ho.
The pinned [ROI configuration](https://github.com/preprocessed-connectomes-project/abide/blob/fda1ec5d3f53eb97d94352e779f37c3ce46caee1/preprocessing/settings/abide_rois.txt)
names ho_mask_pad.nii.gz. It does not by itself identify that historical mask's
checksum or all license/attribution material.

The current [FSL atlas terms](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html)
specifically list Harvard–Oxford under CC BY-SA 4.0, separately from the main FSL
software terms. That current statement should not silently be treated as proof
of the exact historical PCP mask/version or a license for ABIDE participant data.
The staged reference distributes matrix derivatives and stable atlas IDs, not
FSL software or atlas images.

Before changing the reference's redistribution status, complete the remaining
derivative-license scope and acknowledgment record, preserve the relevant
source terms, and review the proposed distribution/use scope. Do not label
the brain dataset MIT or infer unrestricted commercial reuse from the code's
license. This evidence narrows the release review; it is not a public upload
or a claim that all derivative terms have been resolved.

The pinned [PCP overview](https://github.com/preprocessed-connectomes-project/abide/blob/21dae10aaf2b6d576e9f9eee29e48ea6a875daed/index.html)
requests the Craddock et al. 2013 Neuro Bureau Preprocessing Initiative citation
and identifies preprocessing contributors. The [FSL atlas description](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html#harvard-oxford-cortical-and-subcortical-structural-atlases)
supplies atlas credits and references. C-PAC's repository software license is
not a substitute for atlas or participant-data terms. No rights-holder message
has been sent.
