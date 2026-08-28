# svc-rag — Component 3

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python app.py --port 8102
```

Needs `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL` and `OPENAI_EMBED_MODEL` in the
repository-root `.env`. `python tools\check_llm.py` lists what your key can
actually reach — do not guess a model name.

## The corpus problem, and what this does about it

`Work/Nadee/ingest.py` reads `corpus/articles.jsonl`. **That file has never
existed in this repository**, which is why Component 3 had never run end to
end.

1. **It indexes what it reads.** Every article the reader captures is stored
   (`source_type: "read"`), so the corpus builds itself out of use.
2. **A seed corpus if one exists.** `--seed <folder>` loads `.txt` files or a
   `.jsonl` in the shape Nadee's ingest expected.

`POST /forget {"source_type": "read"}` clears what it has remembered.

## What is Nadee's

The prompt, the Sinhala-purity retry, the style/detail word limits, the chunk
metadata and the "always retrieve the current page" rule are hers and are kept
verbatim — see the header of `answer.py`. `Work/Nadee/` is untouched.

What changed is underneath: chroma + langchain + sentence-transformers became
a numpy array and API embeddings. `store.py` says why.

## Note on switching embedding models

The store holds vectors from one model. Changing `OPENAI_EMBED_MODEL` changes
the vector space — delete `store/` before the first run with a new model, or
old vectors will sit alongside new ones and rank meaninglessly.
