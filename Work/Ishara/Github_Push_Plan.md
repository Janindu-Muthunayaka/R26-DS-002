# Pushing to GitHub — analysis and plan

**R26-DS-002 · IT22259134 · 24 August 2026**
**Nothing has been changed yet. This is the plan for your approval.**

---

## 1. What is actually there

I read the repository rather than guessing. Six facts, and two of them change the job completely.

**a) `origin` is the shared team repo.**

```
https://github.com/Janindu-Muthunayaka/R26-DS-002.git
```

`Work/Bumal`, `Work/Janindu` and `Work/Nadee` are your teammates' folders. **Nothing of
theirs will be touched, moved or deleted.** "Only my files" means only your files get
*added* — not that anyone else's get removed.

**b) None of your work has ever been committed.**

22 files are tracked. All 22 are scaffolding:

| what | count |
|---|---|
| `Delete me.txt` placeholders | 4 |
| `t1.txt`, `t2.txt`, `t3.txt` placeholders | 3 |
| empty `__init__.py` | 9 |
| `.gitignore`, `README.md` | 4 |
| `requirements.txt` | 1 |
| `system/README.md` | 1 |

Three commits, all from the folder-structure setup. Every line of code in this system —
`config.py`, `imaging.py`, `layout.py`, `server.py`, the layers, the tools, the tests — is
**untracked**.

**c) Nothing needs deleting.**

This is the important correction to what you asked for. Keeping a file out of GitHub is what
`.gitignore` does, and it does not touch the file on disk. Your `.gitignore` already excludes:

```
models/  *.pt  *.pth  *.safetensors  corpus/  .venv/  __pycache__/
/system/work/  /system/tools/out/  *.zip  *.log
```

So the running system loses nothing. **I will not remove a single file the system reads.**

Worth knowing: the device bridge I work through **cannot delete files on your machine** —
`rm` is blocked by design. Anything to be removed gets moved into a `_to_delete/` folder
(already ignored) and you delete it yourself. That is a safety property, not a limitation.

**d) 85 files are untracked and not ignored — 48 MB.**

About 40 are your real code and documents, a few hundred KB in total. **47 of the 48 MB is
`Work/Ishara/jobs_old/` — 34 phone-capture JPEGs.**

**e) Only three files are genuinely junk.**

| file | what it is |
|---|---|
| `system/app/server.py.bak3` | my backup from the import-path fix |
| `Work/Ishara/71a97929_auto_corrected.txt` | OCR scratch output |
| `.git/index.lock` | **left behind by my read-only `git status`** |

**Delete `.git\index.lock` before you run any git command**, or git will refuse with
"Another git process seems to be running".

**f) The history already carries 80 MB, and it is not yours.**

The `Folder Structure` commit contains `Work/Janindu/0_Data/Model/final_model.pth` (80 MB)
and a whole `venvpaddle` of `.exe` files. They were deleted from the working tree but live in
history forever — that is why `.git` is 80 MB on a repo with 22 tracked files.

**Leave it alone.** Removing them means rewriting history, which breaks every teammate's
clone. Not your problem and not worth the damage.

---

## 2. Your two choices — the honest assessment

You picked Git LFS for the models and "push all 34" for the captures. Both are workable. Here
is what each actually costs, so the decision is informed.

### Git LFS: it works, but it spends Janindu's quota, not yours

Current GitHub limits:

| | value |
|---|---|
| normal push, hard block | **100 MiB per file** |
| normal push, warning | 50 MiB per file |
| LFS storage, Free plan | **10 GiB** |
| LFS bandwidth, Free plan | **10 GiB per month** |
| repository size, GitHub's guidance | ideal < 1 GB, strongly recommended < 5 GB |

`model.safetensors` is **1.15 GB**, so a normal push is impossible — GitHub blocks it at
100 MiB. LFS is the correct mechanism, and 1.2 GB fits inside 10 GiB.

**But the quota belongs to the repository owner.** GitHub's own wording: *"Anyone with write
access to a repository can push files to Git LFS without increasing their personal bandwidth
and storage use."* The bill lands on **Janindu's** account.

What that means in practice:

- 1.2 GB of his 10 GiB storage, permanently.
- 1.2 GB of his monthly bandwidth **every time anyone clones or pulls the model**.
- Four teammates cloning twice ≈ 10 GB — the entire monthly allowance gone. When it runs
  out, LFS pushes and pulls fail **for everyone** until it resets or he buys a data pack.
- Every teammate must run `git lfs install` before cloning. If they do not, they get a
  130-byte pointer file where the model should be, and the system fails in a way that looks
  like a corrupted model.

**So: ask Janindu before doing this.** It is his quota and his repo. If he says yes, I will
set it up properly.

If you would rather not spend his allowance, a **GitHub Release asset** gives the same result
— up to 2 GB per file, free, downloadable by URL, and it never enters git history or LFS
storage. But that is your call; say the word and I do LFS.

### All 34 captures: defensible, and I would have chosen differently

47 MB, under every limit. The real argument for it is good: the captures are the evidence
behind the distance and CER measurements, and a repo that contains them is reproducible.

The cost: every clone is 47 MB bigger, forever — git cannot forget a blob. With the 80 MB
already in history that puts the repo near 130 MB. Still inside GitHub's "ideal under 1 GB",
so nothing breaks.

I would have committed four representative frames and ignored the rest. But your reason is a
real one and the cost is affordable. **I will do it your way.**

---

## 3. The plan

Nothing runs until you say go.

### Phase 0 — you, one command

```
Remove-Item E:\RP\R26-DS-002\.git\index.lock
```

### Phase 1 — tidy, without removing anything the system uses

- Move `system/app/server.py.bak3` and `Work/Ishara/71a97929_auto_corrected.txt` into
  `Work/Ishara/_to_delete/` (already git-ignored). You delete that folder when you like.
- Delete two placeholders whose folders now hold real files: `Work/Ishara/Delete me.txt` and
  `system/web/t3.txt`.
- **Keep** `system/android/t1.txt` and `system/docs/t2.txt` — those folders are still empty,
  and the placeholder is the only thing keeping them in git.
- **Keep** all three teammates' `Delete me.txt`.

### Phase 2 — LFS, only after Janindu agrees

The model currently lives at `E:\RP\corpus\Sinhala_OCR_Correction_v2\models\mt5_plain\`,
**outside the repository**. Git cannot push a file that is not in the repo, so it has to be
copied in — about 1.2 GB more disk.

- Copy the models to `system/models/mt5_plain/` and `system/models/best.pt`.
- **Leave the corpus copy exactly where it is**, and keep running the server with
  `--root E:\RP\corpus\Sinhala_OCR_Correction_v2`. The system today is then completely
  unaffected — it never looks at the repo copy.
- `git lfs install`, then track `*.safetensors` and `*.pt`.
- Add a negation to `.gitignore` so LFS can see the repo copy while everything else stays
  excluded.

### Phase 3 — three commits, so the history reads clearly

1. `system/` — code, layers, tools, tests, web
2. `Work/Ishara/` — documents, ground truth, the 34 captures
3. models, via LFS

### Phase 4 — verify

- `git clone` into a scratch folder and confirm the file list is what we intended and the
  model came down as a real 1.2 GB file, not a pointer.
- Start the server in the **original** folder with the **original** `--root` and confirm it
  still prints `phone should POST to ...`. Nothing about today's working setup may change.

---

## 4. Three things I need from you

1. **Ask Janindu about LFS.** It is his storage and his bandwidth. Yes or no.
2. **Confirm the 1.2 GB copy into the repo.** It is the only way to push a model, and it
   costs 1.2 GB of disk on top of what you have.
3. **The Android app.** It lives at `F:\App`, outside this repo, and `system/android/` holds
   only a placeholder. Push it into this repo, give it its own repo, or leave it out? It is
   your work and it is currently nowhere in version control at all — worth deciding now
   rather than in October.
