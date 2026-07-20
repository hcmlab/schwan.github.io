# ICEP-R MLLM Coding — Public Release

This folder is a self-contained, publish-ready snapshot for the ICMI '26 paper
**"Automated Behavioural Analysis in Parent-Infant Interactions using Multimodal-Large-Language Models"**.
It is meant to become the root of a public GitHub repository (with GitHub Pages enabled on `index.html`).

## Contents

- `index.html` — the project page (GitHub Pages entry point).
- `paper.pdf` — the accepted ICMI '26 paper.
- `assets/` — figures used on the project page.
- `finetuning/` — the `swan_ft` package: profile-driven CLI for dataset building, LoRA
  fine-tuning (via LLaMA-Factory), cross-validation, prediction, reporting, and the
  frozen-feature (DINOv3 + Wav2Vec 2.0) baselines. See `finetuning/README.md`.
- `data-pipeline/` — session discovery, manifest/dataset construction, zero-shot
  annotation generation (Molmo2, Qwen2-Audio, Qwen2.5-Omni), and the ICEP-R
  coding-guideline / prompt documents. See `data-pipeline/README.md`.
- `LICENSE` — GPL-3.0, applies to the code in this repository.

## Publishing checklist

1. **Do not** publish this repo's original git history (the `vm_dev` branch) — it
   contains two now-revoked-should-be Hugging Face tokens in old commits of
   `Tobi-slurm/ft/slurm/common.sh`. This `submission/` folder is a clean export
   with those redacted; start a **fresh** git history from this folder:
   ```bash
   cd submission
   git init
   git add .
   git commit -m "Initial public release"
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. **Rotate/revoke both exposed tokens** at https://huggingface.co/settings/tokens
   if you haven't already — they were committed in cleartext regardless of what
   happens to this export.
3. Before pushing, do a final personal pass over `finetuning/` and `data-pipeline/`
   configs for any cluster paths, usernames, or session identifiers you'd rather
   generalize (several SLURM profile JSONs reference internal mount paths like
   `/mnt/swan/data/...` and lab usernames — not credentials, but you may want to
   genericize them for a fully public repo).
4. Enable GitHub Pages: repo Settings → Pages → Deploy from branch → `main` / `/ (root)`.
5. Update the `href` values in `index.html` (GitHub repo link, DOI) once the repo
   and camera-ready DOI are live.

## What was intentionally excluded from this export

- Raw audio/video recordings and any participant-identifying data (never were
  in the git repo; excluded by design — see the Safe & Responsible Innovation
  statement in the paper).
- SLURM training/eval logs (`Tobi-slurm/ft/logs/`) — large, operational, not
  needed to reproduce the method.
- A stray duplicated `Tobi-slurm/ft/ft/` directory (accidental nested copy in
  the working branch).
- Local `.env` files — only the sanitized `.env.example` templates are included.
