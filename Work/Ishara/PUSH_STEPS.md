# Push to GitHub — the commands

**R26-DS-002 · IT22259134 · 24 August 2026**

Everything that could be prepared without git has been done already:

- `.gitattributes` written — LFS scoped to `system/models/` only, so no
  teammate's future file is swept into LFS
- root `.gitignore` and `system/.gitignore` — exceptions added so `system/models/`
  is visible to git while every other model path stays ignored. Verified with
  `git check-ignore`.
- `system/models/best.pt` copied in (40.5 MB)
- `system/models/mt5_plain/` — config, generation_config, tokenizer,
  tokenizer_config copied in
- `system/models/PUT_MODELS_HERE.md` rewritten with LFS clone instructions
- junk moved to `Work/Ishara/_to_delete/` (git-ignored; delete it whenever)

**Not done, because I could not:** `model.safetensors` is 1.15 GB and the file
bridge moves about 5 MB/s, so the copy would take four minutes and time out.
You do it locally in step 2 — it takes seconds.

**Also not done:** anything that writes to git. The bridge cannot remove
`.git/index.lock`, so every git command I run leaves one behind. All git runs
from your PowerShell.

---

## 1. Clear the stale lock

```
Remove-Item E:\RP\R26-DS-002\.git\index.lock -ErrorAction SilentlyContinue
```

## 2. Copy the big model in

```
Copy-Item "E:\RP\corpus\Sinhala_OCR_Correction_v2\models\mt5_plain\model.safetensors" "E:\RP\R26-DS-002\system\models\mt5_plain\model.safetensors"
```

Check it arrived whole — must print **1200729512**:

```
(Get-Item E:\RP\R26-DS-002\system\models\mt5_plain\model.safetensors).Length
```

## 3. Turn LFS on for this repo

```
cd E:\RP\R26-DS-002
```
```
git lfs install
```
```
git lfs track
```

The last one should list the three patterns from `.gitattributes`. If it prints
nothing, `.gitattributes` is not being read — stop and tell me.

## 4. Drop two placeholders that are no longer needed

```
git rm "Work/Ishara/Delete me.txt" system/web/t3.txt
```

`system/android/t1.txt` and `system/docs/t2.txt` stay — those folders are still
empty and the placeholder is the only thing keeping them in git. All three
teammates' `Delete me.txt` files stay.

## 5. Get your teammates' work first

```
git fetch origin
```
```
git pull --rebase origin main
```

Everything you are adding is in paths nobody else touches, so a conflict is very
unlikely. If the rebase does stop, run `git status`, send me what it says, and
**do not** run `git rebase --skip` — that throws work away.

## 6. Commit in three parts

**The system:**

```
git add .gitattributes .gitignore system/.gitignore setup_project.py system/app system/core system/layers system/tests system/tools system/web
```
```
git commit -m "Component 2: close-up capture pipeline, layout analysis, tools and tests"
```

**Your documents and the captures:**

```
git add Work/Ishara
```
```
git commit -m "Ishara: measurement records, ground truth, and the phone captures behind them"
```

**The models, via LFS** (this one is slow — 1.2 GB has to upload):

```
git add system/models
```
```
git commit -m "Models: mT5 post-OCR corrector and YOLO11m article detector (Git LFS)"
```

Before pushing, confirm the two big files really went to LFS:

```
git lfs ls-files
```

You should see `best.pt` and `model.safetensors`. **If that command prints
nothing, do not push** — it means the 1.2 GB is about to go into plain git,
GitHub will reject it at 100 MiB, and the commit will need to be undone.

## 7. Push

```
git push origin main
```

The LFS upload is the slow part. On a normal connection expect several minutes
for 1.2 GB. If it fails part-way, run the same command again — LFS resumes.

## 8. Check nothing broke

Your working system is untouched by all of the above. Confirm it:

```
cd E:\RP\R26-DS-002\system
```
```
python -m app.server --root E:\RP\corpus\Sinhala_OCR_Correction_v2
```

It should still print `phone should POST to http://...`. Nothing in `core/` or
`layers/` was edited and `--root` still points at the corpus folder, so this is
the same system that was running an hour ago.

Then, as a separate check that the push is actually usable:

```
cd $env:TEMP
```
```
git clone https://github.com/Janindu-Muthunayaka/R26-DS-002.git clonecheck
```
```
(Get-Item clonecheck\system\models\mt5_plain\model.safetensors).Length
```

**1200729512** means LFS worked. A few hundred bytes means the clone got a
pointer file — recoverable with `git lfs install; git lfs pull`, and it is
exactly what your teammates will hit if they skip `git lfs install`.

---

## What is being pushed

| | size |
|---|---|
| code, tools, tests, web | a few hundred KB |
| `Work/Ishara/` documents and ground truth | ~200 KB |
| `Work/Ishara/jobs_old/` — 34 phone captures | 47 MB |
| `system/models/mt5_plain/tokenizer.json` | 16 MB (plain git, under the limit) |
| `system/models/best.pt` | 40.5 MB (**LFS**) |
| `system/models/mt5_plain/model.safetensors` | 1.15 GB (**LFS**) |

Not pushed, and still on your disk: `.venv/`, `__pycache__/`, `system/work/`,
`system/tools/out/`, the corpus, and the corpus copies of the models.

## Two things worth telling the team

1. **`git lfs install` before cloning**, or the models come down as pointer
   files. `system/models/PUT_MODELS_HERE.md` says so, but it is worth saying in
   the group chat too.
2. **LFS storage and bandwidth are billed to Janindu**, as repository owner —
   10 GiB of each per month on the free plan. The model uses 1.2 GB of storage
   and 1.2 GB of bandwidth per clone. Four people cloning twice is most of the
   monthly allowance. Nobody should add more large files without asking him.
