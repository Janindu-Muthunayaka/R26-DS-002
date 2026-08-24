# Models

These files are in the repository, stored with **Git LFS**.

    system/models/
        best.pt                      40.5 MB   YOLO11m article detector
        mt5_plain/
            config.json
            generation_config.json
            model.safetensors        1.15 GB   the post-OCR corrector
            tokenizer.json           16.3 MB
            tokenizer_config.json

## Cloning — read this or the system will fail confusingly

Run **once per machine, before you clone**:

    git lfs install

Then clone normally. If you clone without it, `best.pt` and `model.safetensors`
arrive as ~130-byte text files that begin `version https://git-lfs...`. The
pipeline then fails while loading the model, and the error looks like a
corrupted checkpoint rather than a missing download.

Already cloned without LFS? Fix it in place:

    git lfs install
    git lfs pull

## Check they are real

    # PowerShell
    (Get-Item system\models\mt5_plain\model.safetensors).Length   # 1200729512
    (Get-Item system\models\best.pt).Length                       # 40531820

A few hundred bytes means you have a pointer, not a model.

## Running

    cd system
    python -m app.server --root <folder that contains models\ >

`--root` is the folder holding `models/` and `layout/runs/...`, not the
repository root. Two roots work:

    --root E:\RP\R26-DS-002\system              (this checkout)
    --root E:\RP\corpus\Sinhala_OCR_Correction_v2   (the local corpus copy)

The corpus copy is the one the system has been developed against and it is
deliberately left outside the repository. Both contain the same weights.

## Why LFS and not plain git

GitHub blocks any single file over 100 MiB on a normal push, and
`model.safetensors` is 1.15 GB. LFS storage and bandwidth are billed to the
**repository owner**, not to whoever pushes — so do not add more large files
here without asking. `.gitattributes` deliberately scopes the LFS rules to
`system/models/` only, so nothing else in the repo is swept into LFS.
