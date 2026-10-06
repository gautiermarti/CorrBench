# ABIDE / PCP attribution for the CorrBench brain cohorts

We thank ABIDE I participants and families, the contributing investigators,
FCP/INDI, the Preprocessed Connectomes Project and the C-PAC team.
This record identifies the source datasets actually used, with counts
verified against each hashed matrix split. It is not a rights clearance.
The cohorts overlap: 601 + 601 + 762 is not an independent sample count.

## Sources, contributions and changes

ABIDE: Di Martino et al. (2014), *Molecular Psychiatry* 19, 659–667,
[doi:10.1038/mp.2013.78](https://doi.org/10.1038/mp.2013.78).
PCP: Craddock et al. (2013), *The Neuro Bureau Preprocessing Initiative:
Open Sharing of Preprocessed Neuroimaging Data and Derivatives*,
[doi:10.3389/conf.fninf.2013.09.00041](https://doi.org/10.3389/conf.fninf.2013.09.00041).
The PCP overview credits C-PAC/ANTS processing to Daniel Clark, Zarrar
Shehzad, Qingyang Li, Steven Giavasis, John Pellman and R. Cameron Craddock.
PCP founders are R. Cameron Craddock and Pierre Bellec. Full project
credits remain at the [pinned overview](https://github.com/preprocessed-connectomes-project/abide/blob/21dae10aaf2b6d576e9f9eee29e48ea6a875daed/index.html).

Inputs are PCP `cpac/filt_noglobal/rois_ho` regional series, not raw MRI.
CorrBench applies documented QC, per-scan standardization, Pearson/OAS
estimation and subject/site splitting. Original111 preserves the original
mask, including unnamed two-voxel ID 3455. The declared named110 variants
remove that ID before standardization and re-estimation. These are
CorrBench transformations, not unmodified ABIDE or official PCP releases.

Harvard–Oxford attribution: the atlas is provided by the FSL project;
its [official description](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html#harvard-oxford-cortical-and-subcortical-structural-atlases)
credits the Harvard Center for Morphometric Analysis and collaborating
investigators. The separate [atlas audit](abide_atlas_audit_20261006.md)
pins the actual PCP mask. No FSL software or atlas image is bundled.
Atlas contributors include David Kennedy, Christian Haselgrove, Bruce
Fischl, Janis Breeze, Jean Frazier, Larry Seidman and Jill Goldstein.
The atlas page credits NIH P41-RR14075, R01 RR16594-01A1, R01 NS052585-01,
K08 MH01573 and K01 MH01798 for the source segmentation work.

## ABIDE coordination funding

The ABIDE source page credits NIMH K23MH087770 and the Leon Levy
Foundation for Adriana Di Martino's work; Joseph P. Healy and Stavros
Niarchos Foundation gifts to the Child Mind Institute, and NIMH
R03MH096321, for Michael P. Milham and the INDI team. These supported
the source resource; they are not claimed as grants to CorrFlow.

## Included datasets and funding

Funding and investigator names below come from the [ABIDE I profiles](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html).
Counts are original111 / named110 same601 / named110 expanded762.
Source grant identifiers and spellings are retained; `None` is the
source's statement, not a missing parser result. CMU contributes only
to the expanded762 cohort.

### CALTECH — California Institute of Technology

Included scans: 1 / 1 / 11.

Investigators: Daniel P. Kennedy, Ph.D., J. Michael Tyszka, Ph.D., Ralph Adolphs, Ph.D.

Source funding: Simons Foundation (SFARI-07-01 to R.A.) National Institute of Mental Health (R01 MH080721 to R.A.)

### CMU — Carnegie Mellon University

Included scans: 0 / 0 / 1.

Investigators: Marlene Behrmann, Ph.D., Ilan Dinstein, Ph.D., Nancy Minshew, M.D.

Source funding: NICHD/NIDCD PO1/U19 to M. B. (PI: Nancy Minshew), which is part of the NICHD/NIDCD Collaborative Programs for Excellence in Autism Simons Foundation to M. B. (PI: David Heeger)

### KKI — Kennedy Krieger Institute

Included scans: 26 / 26 / 29.

Investigators: Stewart H. Mostofsky, M.D., Mary Beth Nebel, Ph.D.

Source funding: Autism Speaks Foundation (Mostofsky) National Institute of Neurological Disorders and Stroke: R01 NS048527 (Mostofsky)

### MAX_MUN — Ludwig Maximilians University Munich

Included scans: 19 / 19 / 40.

Investigators: Sophia Mueller, M.D., Birgit Ertl-Wagner, M.D., Kristina Hennig-Fast, Michele Noterdaeme, M.D.

Source funding: None

### NYU — NYU Langone Medical Center

Included scans: 160 / 160 / 166.

Investigators: Adriana Di Martino, M.D., F. Xavier Castellanos, M.D., Michael P. Milham, M.D., Ph.D., Clare Kelly, Ph.D.

Source funding: NIH (K23MH087770; R21MH084126; R01MH081218; R01HD065282) Autism Speaks The Stavros Niarchos Foundation The Leon Levy Foundation An endowment provided by Phyllis Green and Randolph Co̅wen

### OLIN — Olin, Institute of Living at Hartford Hospital

Included scans: 15 / 15 / 22.

Investigators: Michal Assaf, M.D.

Source funding: Autism Speaks (to M.A.). Hartford Hospital (to M.A)

### OHSU — Oregon Health and Science University

Included scans: 14 / 14 / 22.

Investigators: Damien Fair, PA-C, Ph.D., Joel Nigg, Ph.D.

Source funding: R00 MH091238 (Fair) R01 MH096773 (Fair) R01 MH086654 (Nigg) Simon Foundation, Inc. (Nigg)

### SDSU — San Diego State University

Included scans: 6 / 6 / 20.

Investigators: Ralph-Axel Müller, Ph.D., Christopher Keown, B.S.

Source funding: NIH R01-MH081023 (PI: Müller)

### SBL — Social Brain Lab BCN NIC UMC Groningen and Netherlands Institute for Neurosciences

Included scans: 21 / 21 / 22.

Investigators: Marc Thioux, Ph.D., Leonardo Cerliani, Ph.D., Christian Keysers, Ph.D.

Source funding: Nederlandse Organisatie voor Wetenschappelijk Onderzoek (NWO) Cognitive Pilot Project (051.07.003) NWO VIDI (452-04-305) NWO Open Competition (400-08-089) European Commission, Marie Curie Excellence Grant (MEXT-CT-2005-023253) National Initiative for Brain and Cognition NIHC HCMI Functional Markers (056-13-014) National Initiative for Brain and Cognition NIHC HCMI Simulation and TOM (056-13-017) Netherlands Brain Foundation (KS 2010(1)-29)

### STANFORD — Stanford University

Included scans: 23 / 23 / 24.

Investigators: Lucina Uddin, Ph.D., Kaustubh Supekar, Ph.D., Vinod Menon, Ph.D.

Source funding: National Institute of Mental Health Career Development Award [K01MH092288] to L.U. Stanford Institute for Neuro-Innovation & Translational Neurosciences National Institutes of Health [DC011095, MH084164] to V.M.

### TRINITY — Trinity Centre for Health Sciences

Included scans: 18 / 18 / 41.

Investigators: Louise Gallagher, MB, MRCPsych, Ph.D., Sonja Delmonte, Joshua Balsters, Ph.D.

Source funding: The Meath Foundation, Adelaide and Meath Hospital, incorporating the National Children's Hospital (AMNCH), Tallaght, and travel fellowship by the Kyulan Family Foundation.

### UCLA_1 — University of California, Los Angeles: Sample 1

Included scans: 46 / 46 / 52.

Investigators: Mirella Dapretto, Ph.D., Susan Bookheimer, Ph.D., Jeffrey D. Rudie

Source funding: UCLA Autism Center of Excellence NICHD P50 HD055784 NIMH 1R01 HD065280-01

### UCLA_2 — University of California, Los Angeles: Sample 2

Included scans: 14 / 14 / 15.

Investigators: Susan Y. Bookheimer, Ph.D., Mirella Dapretto, Ph.D., Jeffrey D. Rudie

Source funding: UCLA Autism Center of Excellence NICHD P50 HD055784 NIMH 1R01 HD065280-01

### LEUVEN_1 — University of Leuven: Sample 1

Included scans: 20 / 20 / 28.

Investigators: Kaat Alaerts, Ph.D., Nicole Wenderoth, Ph.D.

Source funding: Flanders Fund for Scientific Research (FWO projects G.0758.10) FWO postdoctoral Research fellowship grant (KA) Grant P6/29 from the Interuniversity Attraction Poles program of the Belgian federal government. This study has been conducted in collaboration with the Leuven Autism Research Consortium (LAuRes), funded by the Research Council of the University of Leuven (IDO/08/013).

### LEUVEN_2 — University of Leuven: Sample 2

Included scans: 25 / 25 / 26.

Investigators: Stefan Sunaert, M.D., Ph.D., Judith Verhoeven

Source funding: Fund for Scientific Research-Flanders (F.W.O.) (research grant G.0354.06, doctoral mandate to Judith Verhoeven, research grant 1841313N, senior clinical investigator grant to Stefan Sunaert) Belgian Inter University Attraction Pole (grant 6/29) KU Leuven Research Council (grant IDO/08/013)

### UM_1 — University of Michigan: Sample 1

Included scans: 63 / 63 / 77.

Investigators: Christopher S. Monk, Ph.D., Jillian Lee Wiggins, M.S., Scott Peltier, Ph.D., Catherine Lord, Ph.D.

Source funding: Autism Speaks (CSM) NIH U19 HD035482 and NIH MH066496 (CL) Autism Speaks Pre-doctoral Fellowship 4773 (JLW) Michigan Institute for Clinical and Health Research (MICHR) Pre-doctoral Fellowship UL1RR024986 (JLW) NIH R21 MH079871 (SP)

### UM_2 — University of Michigan: Sample 2

Included scans: 29 / 29 / 31.

Investigators: Scott Peltier, Ph.D., Catherine Lord, Ph.D., Christopher S. Monk, Ph.D., Jillian Lee Wiggins, M.S., Susan Risi, Ph.D., Opal Ousley, Ph.D.

Source funding: Autism Speaks (CM) NIH U19 HD035482 and MH066496 (CL) Autism Speaks Pre-doctoral Fellowship 4773 (JW) Michigan Institute for Clinical and Health Research (MICHR) Pre-doctoral Fellowship UL1RR024986 (JW) NIH R21 MH079871 (SP)

### PITT — University of Pittsburgh School of Medicine

Included scans: 31 / 31 / 41.

Investigators: Beatriz Luna, Ph.D., Kirsten O'Hearn, Ph.D.

Source funding: Autism Speaks Grant 04593 (BL) KO1 NIMH MH081191 (KO) NIMH MH67924 (BL) NIH HD55748 (NM)

### USM — University of Utah School of Medicine

Included scans: 44 / 44 / 59.

Investigators: Jeffrey S. Anderson, M.D., Ph.D., Jared A. Nielsen, B.S., Alyson L. Froehlich, Ph.D., Molly B. DuBray, M.A., Michael A. Ferguson, Ph.D., T. Jason Druzgal, M.D., Ph.D., Annahir N. Cariello, Jason R. Cooperrider, B.S., Brandon A. Zielinski, M.D., Ph.D., Caitlin Ravichandran, Ph.D., P. Thomas Fletcher, Ph.D., Andrew L. Alexander, Ph.D., Erin D. Bigler, Ph.D., Nicholas Lange, Sc.D., Janet E. Lainhart, M.D.

Source funding: National Institutes of Health (grant numbers: K08 MH092697, RO1MH080826, P50MH60450, T32DC008553, R01NS34783) Autism Speaks Mentor-based Predoctoral Fellowship (grant number: 1677) University of Utah Multidisciplinary Research Seed Grant NRSA Predoctoral Fellowship (grant number: F31 DC010143) Ben B. and Iris M. Margolis Foundation

### YALE — Yale Child Study Center

Included scans: 26 / 26 / 35.

Investigators: Ben Deen, Kevin Pelphrey, Ph.D.

Source funding: Simons Foundation (KP) Autism Speaks (KP) John Merck Scholars Fund (KP) Autism Science Foundation NICHD (KP) NIMH

## Data terms

The [ABIDE usage page](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) links [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). Any redistribution must retain applicable attribution, change notices, noncommercial and share-alike conditions. The software MIT license does not replace data terms. The separate FSL Harvard–Oxford terms and exact derivative scope remain recorded in [the release review](abide_release_review.md). No endorsement by the source investigators or funders is implied.
