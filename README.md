# schwan.github.io

Project page for the DFG SCHWAN project (project number 490909448): automated ICEP-R
behavioral coding of parent-infant interaction in the Face-to-Face Still-Face paradigm.

Live site (once GitHub Pages is enabled, Settings → Pages → Deploy from branch `main` / root):
`https://hcmlab.github.io/schwan.github.io/`

## Structure

```
/
├── index.html       — SCHWAN project landing page (publication timeline, overview)
├── 2026-icmi/        — ICMI '26 publication: project page + full fine-tuning codebase
│                        (mirrors https://github.com/Daksitha/schwan-finetune)
├── 2024-icmi/        — ICMI '24 publication: project page linking out to
│                        https://github.com/Daksitha/SCHWAN-ICEP-R-Automation
└── LICENSE
```

`2024-icmi/` intentionally does not vendor a copy of that repository's code — it links out to
keep a single source of truth. `2026-icmi/` contains a full, secret-scrubbed snapshot of the
ICMI '26 codebase so it is directly browsable on the live Pages site; the canonical, actively
developed copy remains at the linked GitHub repository.

## Updating

- **This year's page/code** (`2026-icmi/`): edit in the `schwan-finetune` repo's `submission/`
  folder, then copy the updated contents over `2026-icmi/` here and commit.
- **Landing page** (`index.html`): update the publication timeline when a new paper is added,
  following the same `YYYY-venue/` folder convention.
