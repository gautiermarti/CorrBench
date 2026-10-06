# CorrBench competition prototype

This directory prepares a Kaggle Community Competition locally. Nothing has
been created on Kaggle. The current bundle uses **entirely synthetic data** to
test the submission contract; it is not a new benchmark result.

## Submission contract

Each requested ID receives one symmetric matrix with unit diagonal, encoded as
the strict lower triangle in NumPy row-major lower-triangle order. At dimension
80 this is 3,160 numerical columns, named c_0000 through c_3159. The public case
file supplies the condition. Participants submit ID plus entries only; they
cannot change conditions. IDs must match exactly. Non-numeric, missing, infinite,
out-of-range or indefinite matrices are rejected. The score never silently
repairs them. PSD tolerance is 1e-6; logarithmic features clip eigenvalues at
1e-12, both explicit candidate-metric choices.

The candidate score compares **distributions within conditions**, not arbitrary
generated/real row pairs. It averages 128 one-dimensional Wasserstein-2 distances
of 21 standardized entry/spectrum summaries, then averages equally across the
three conditions. Training data alone fit feature scales; a fixed NumPy random
seed defines directions. This is a separately versioned NumPy float64 metric,
not the exact Torch score used in the frozen paper. Lower is better.

Kaggle's solution Usage field determines Public/Private subsets. Each subset
must independently contain at least 20 cases per condition. Group assignment
must precede row assignment; all observations from the same protected independent
unit belong to one subset. The local synthetic fixture has 200 independent
groups, each with three cases, with 40 Public and 160 Private groups.

## Local commands

From the repository root:

~~~bash
python research/build_kaggle_fixture.py --output artifacts/kaggle_fixture
python competition/starter_replay.py \
  --data artifacts/kaggle_fixture/public --output artifacts/replay.csv
python -m pytest tests/test_competition_metric.py
~~~

The generated public directory contains train.csv, test.csv,
sample_submission.csv, a standalone configured metric.py and metric.ipynb, and
starter.ipynb. The organizer_only directory contains solution.csv and group
membership. Never upload organizer files as participant data. The reference
perfect submission is used only to test scoring, not as a baseline.

The starter reads only public training matrices and requested conditions.
The builder exercises CSV parsing, both scoring subsets, row-order invariance,
within-condition distribution invariance, perfect score zero, replay, mean
collapse, identity and incorrect conditions. Separate tests reject pathological
submissions. Audit files report source/data hashes, runtime and memory.

To execute the generated starter and metric notebooks in the same environment:

~~~bash
python -m pip install -e '.[notebooks]'
python -m ipykernel install --sys-prefix --name corrbench-readiness
python research/check_competition_notebooks.py --fixture artifacts/kaggle_fixture \
  --output artifacts/executed_notebooks --kernel corrbench-readiness
~~~

The executed organizer metric notebook contains hidden-fixture validation code;
keep that output private. The generated public metric notebook remains separate.

## What still has to be decided before launch

1. A protected fresh holdout with sufficient independent units and explicit
   source rights. Existing public returns and published matrix releases can
   reconstruct targets and cannot become a secret competition test by renaming
   rows. Synthetic fixture seeds and distributions are public by design.
2. A final metric chosen through adversarial and utility studies, including
   degeneracy, replay, sample count, feature scale, group uncertainty and seed
   sensitivity. Twenty-one summaries cannot identify the full distribution.
   Anonymous financial structure is not suitable for judging named brain regions
   or geographic locations; those require domain-specific functionals.
3. Private/public split policy, submission limits, compute budget, eligibility,
   prize/rules text, data terms and finalist code-rerun procedure. A CSV leaderboard
   alone does not establish a reusable stochastic generator.
4. An unpublished hosted Kaggle test. Run known perfect, malformed and baseline
   submissions through the actual metric sandbox, check identical public/private
   behavior, and measure runtime and memory on its hardware. Local passing tests
   do not certify hosted compatibility.
5. An approved launch and coordinated public code/data links.

The annotated score(solution, submission, row_id_column_name) function,
ParticipantVisibleError class, no-network metric and Usage design follow the
[official competition setup documentation](https://www.kaggle.com/docs/competitions-setup).
That documentation currently specifies a 30-minute custom-metric runtime limit;
the hosted test is still required. Metrics and starter notebooks are generated
from reviewable source, with configuration fitted before any evaluation.
