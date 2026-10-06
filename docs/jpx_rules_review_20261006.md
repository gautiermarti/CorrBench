# JPX competition source-use review

Checked 2026-10-06. Status: **restricted**, with separate research/publication
permission not established. This is a release decision based on the posted
source terms, not a legal determination about every aggregate research result.

The [official competition rules](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/rules)
rendered in ordinary logged-out Chromium, HTTP 200, at the same URL. No rules
were accepted and no competition data were downloaded. The page labels data
access **“COMPETITION USE ONLY”**. Section 7.A limits use to participation and
Kaggle forums; section 7.B prohibits sharing the competition data with
nonparticipants. The winner open-source requirement concerns submissions/code;
it is not a blanket dataset license. The [dataset page](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data)
also refers users to the competition rules.

The captured body has 31,865 characters and SHA-256
`cd8f42acd9d315b732eca6203d3e91e03cc608097008bd24f8ba0b4375a09839`.
The local research evidence is under
`artifacts/completion_20261006/financial_terms/jpx_rules_browser.{txt,html,json}`.
Full terms are retained locally, not reproduced in the public companion.
The [machine-readable review](jpx_rules_review_20261006.json) records this check.

The preserved source ZIP has SHA-256
`aa144566f63e62a5af9ac5a7851e42ded46554fbfe650f851223b01920210c85`.
Its 24 members contain no separate README, license or terms file. The 8,672
derived matrix records and all existing numerical results remain unchanged.

## Consequences for the current release

- Keep Japan arrays local and set `rights_status: restricted`; no open data
  license or HF upload is asserted. Do not run new Japan experiments while the
  separate research-use permission is unresolved.
- Mark the Japan appendix in both review drafts with the source-use issue.
  Its unfavorable replay comparison and split-overlap limitations are preserved.
- Establish whether the author already has an applicable separate grant, and
  its scope for research use, publication, derived matrices and model weights.
  If needed, the [permission question](jpx_permission_question_draft.md) is ready
  for author review; it is unsent.
- Do not infer a permission grant from downloadable data, code licensing,
  transformed matrices, or a previous inconclusive browser check. A general JPX
  market-data agreement is not evidence of this competition's grant.

The immutable rc1 software/teaching packet predates this finding. It contains
no financial source arrays and remains a valid synthetic teaching fixture;
its older source-review prose is superseded by this record. Current exporter
changes are source-checkout changes, not a new installed rc1 release.
