"""Bind ABIDE site/funder credits to hashed matrix cohorts; no rights clearance."""

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

SITE_NAMES = {
    "CALTECH": "California Institute of Technology",
    "CMU": "Carnegie Mellon University",
    "KKI": "Kennedy Krieger Institute",
    "MAX_MUN": "Ludwig Maximilians University Munich",
    "NYU": "NYU Langone Medical Center",
    "OLIN": "Olin, Institute of Living at Hartford Hospital",
    "OHSU": "Oregon Health and Science University",
    "SDSU": "San Diego State University",
    "SBL": (
        "Social Brain Lab BCN NIC UMC Groningen and Netherlands Institute "
        "for Neurosciences"
    ),
    "STANFORD": "Stanford University",
    "TRINITY": "Trinity Centre for Health Sciences",
    "UCLA_1": "University of California, Los Angeles: Sample 1",
    "UCLA_2": "University of California, Los Angeles: Sample 2",
    "LEUVEN_1": "University of Leuven: Sample 1",
    "LEUVEN_2": "University of Leuven: Sample 2",
    "UM_1": "University of Michigan: Sample 1",
    "UM_2": "University of Michigan: Sample 2",
    "PITT": "University of Pittsburgh School of Medicine",
    "USM": "University of Utah School of Medicine",
    "YALE": "Yale Child Study Center",
}
SOURCE = "https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html"


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(text):
    parser = PlainText()
    parser.feed(re.sub(r"<sup>.*?</sup>", "", text, flags=re.S))
    return " ".join(" ".join(parser.parts).split())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section(chunk, title):
    match = re.search(
        rf"<h3[^>]*>\s*{title}\s*</h3>(.*?)(?:<h3|$)", chunk, re.S
    )
    if not match:
        raise ValueError(f"Missing source section: {title}")
    text = plain(match[1])
    if not text:
        raise ValueError(f"Empty source section: {title}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--same601", type=Path, required=True)
    parser.add_argument("--expanded762", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    provenance = json.loads((args.terms / "manifest.json").read_text())
    source = args.terms / "abide_i.html"
    if sha(source) != provenance["sources"][source.name]["sha256"]:
        raise ValueError("Source terms changed after retrieval")
    document = source.read_text()
    sections = {}
    for chunk in re.split(r'<div\s+class="site">', document)[1:]:
        title = re.search(r"<h[24][^>]*>(.*?)</h[24]>", chunk, re.S)
        if not title:
            raise ValueError("Unrecognized site heading")
        name = plain(title[1])
        if name in sections:
            raise ValueError("Repeated site heading")
        sections[name] = {
            "investigators": section(chunk, "Investigators"),
            "funding_as_published": section(chunk, "Funding"),
        }
    if set(sections) != set(SITE_NAMES.values()):
        raise ValueError("Source site headings changed; review the mapping")
    cohorts = {}
    for name, path, count in [
        ("original111", args.original, 601),
        ("named110_same601", args.same601, 601),
        ("named110_expanded762", args.expanded762, 762),
    ]:
        manifest = json.loads((path / "manifest.json").read_text())
        splits, subjects = {}, set()
        for split in ["train", "validation", "test"]:
            file = path / f"{split}.npz"
            expected = manifest["splits"][split]
            if sha(file) != expected.get("sha256", expected.get("source_sha256")):
                raise ValueError(f"Changed frozen source: {name}/{split}")
            with np.load(file, allow_pickle=False) as arrays:
                values, counts = np.unique(arrays["site_ids"], return_counts=True)
                sites = dict(zip(values.tolist(), counts.tolist(), strict=True))
                ids = set(arrays["subject_ids"].tolist())
                if ids & subjects or len(ids) != len(arrays["subject_ids"]):
                    raise ValueError("Repeated subject within or across splits")
                subjects |= ids
                if not set(sites) <= set(SITE_NAMES):
                    raise ValueError("Site missing from attribution mapping")
            splits[split] = {
                "subjects": sum(sites.values()), "sites": sites,
                "npz_sha256": sha(file),
            }
        if len(subjects) != count:
            raise ValueError(f"Unexpected cohort size: {name}")
        cohorts[name] = {
            "subjects": count, "manifest_sha256": sha(path / "manifest.json"),
            "splits": splits,
        }
    records = {}
    for code, name in SITE_NAMES.items():
        records[code] = {
            "institution": name, **sections[name],
            "source": SOURCE,
            "cohort_counts": {
                cohort: sum(s["sites"].get(code, 0) for s in item["splits"].values())
                for cohort, item in cohorts.items()
            },
        }
    report = {
        "schema": "corrbench.abide-attribution.v1",
        "date": "2026-10-06", "rights_clearance": False,
        "source_terms": provenance,
        "cohorts": cohorts, "sites": records,
        "limits": [
            "The cohorts overlap; their counts must not be added as "
            "independent subjects",
            "Attribution completion does not settle derivative license scope",
            "Funding strings preserve source spellings and grant identifiers",
            "No raw images, atlas images or regional time series are distributed here",
        ],
    }
    lines = [
        "# ABIDE / PCP attribution for the CorrBench brain cohorts", "",
        "We thank ABIDE I participants and families, the contributing investigators,",
        "FCP/INDI, the Preprocessed Connectomes Project and the C-PAC team.",
        "This record identifies the source datasets actually used, with counts",
        "verified against each hashed matrix split. It is not a rights clearance.",
        "The cohorts overlap: 601 + 601 + 762 is not an independent sample count.",
        "",
        "## Sources, contributions and changes", "",
        "ABIDE: Di Martino et al. (2014), *Molecular Psychiatry* 19, 659–667,",
        "[doi:10.1038/mp.2013.78](https://doi.org/10.1038/mp.2013.78).",
        "PCP: Craddock et al. (2013), *The Neuro Bureau Preprocessing Initiative:",
        "Open Sharing of Preprocessed Neuroimaging Data and Derivatives*,",
        "[doi:10.3389/conf.fninf.2013.09.00041](https://doi.org/10.3389/conf.fninf.2013.09.00041).",
        "The PCP overview credits C-PAC/ANTS processing to Daniel Clark, Zarrar",
        "Shehzad, Qingyang Li, Steven Giavasis, John Pellman and R. Cameron Craddock.",
        "PCP founders are R. Cameron Craddock and Pierre Bellec. Full project",
        "credits remain at the [pinned overview](https://github.com/",
        "preprocessed-connectomes-project/abide/blob/",
        "21dae10aaf2b6d576e9f9eee29e48ea6a875daed/index.html).", "",
        "Inputs are PCP `cpac/filt_noglobal/rois_ho` regional series, not raw MRI.",
        "CorrBench applies documented QC, per-scan standardization, Pearson/OAS",
        "estimation and subject/site splitting. Original111 preserves the original",
        "mask, including unnamed two-voxel ID 3455. The declared named110 variants",
        "remove that ID before standardization and re-estimation. These are",
        "CorrBench transformations, not unmodified ABIDE or official PCP releases.",
        "",
        "Harvard–Oxford attribution: the atlas is provided by the FSL project;",
        "its [official description](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/",
        "datasets.html#harvard-oxford-cortical-and-subcortical-structural-atlases)",
        "credits the Harvard Center for Morphometric Analysis and collaborating",
        "investigators. The separate [atlas audit](abide_atlas_audit_20261006.md)",
        "pins the actual PCP mask. No FSL software or atlas image is bundled.",
        "Atlas contributors include David Kennedy, Christian Haselgrove, Bruce",
        "Fischl, Janis Breeze, Jean Frazier, Larry Seidman and Jill Goldstein.",
        "The atlas page credits NIH P41-RR14075, R01 RR16594-01A1, R01 NS052585-01,",
        "K08 MH01573 and K01 MH01798 for the source segmentation work.",
        "",
        "## ABIDE coordination funding", "",
        "The ABIDE source page credits NIMH K23MH087770 and the Leon Levy",
        "Foundation for Adriana Di Martino's work; Joseph P. Healy and Stavros",
        "Niarchos Foundation gifts to the Child Mind Institute, and NIMH",
        "R03MH096321, for Michael P. Milham and the INDI team. These supported",
        "the source resource; they are not claimed as grants to CorrFlow.",
        "",
        "## Included datasets and funding", "",
        "Funding and investigator names below come from the "
        f"[ABIDE I profiles]({SOURCE}).",
        "Counts are original111 / named110 same601 / named110 expanded762.",
        "Source grant identifiers and spellings are retained; `None` is the",
        "source's statement, not a missing parser result. CMU contributes only",
        "to the expanded762 cohort.", "",
    ]
    # Join URLs split above for source-line readability.
    text = "\n".join(lines).replace(
        "https://github.com/\npreprocessed-connectomes-project/abide/blob/\n",
        "https://github.com/preprocessed-connectomes-project/abide/blob/",
    ).replace("https://fsl.fmrib.ox.ac.uk/fsl/docs/other/\n", "https://fsl.fmrib.ox.ac.uk/fsl/docs/other/")
    for code, item in records.items():
        counts = " / ".join(str(x) for x in item["cohort_counts"].values())
        text += (
            f"\n### {code} — {item['institution']}\n\n"
            f"Included scans: {counts}.\n\n"
            f"Investigators: {item['investigators']}\n\n"
            f"Source funding: {item['funding_as_published']}\n"
        )
    text += (
        "\n## Data terms\n\n"
        "The [ABIDE usage page](" + SOURCE + ") links "
        "[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). "
        "Any redistribution must retain applicable attribution, change notices, "
        "noncommercial and share-alike conditions. The software MIT license "
        "does not replace data terms. The separate FSL Harvard–Oxford terms "
        "and exact derivative scope remain recorded in "
        "[the release review](abide_release_review.md). No endorsement by "
        "the source investigators or funders is implied.\n"
    )
    args.output.mkdir(parents=True)
    (args.output / "abide_attribution_20261006.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output / "abide_attribution_20261006.md").write_text(text)
    print(json.dumps({"sites": len(records), "cohorts": {
        k: v["subjects"] for k, v in cohorts.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
