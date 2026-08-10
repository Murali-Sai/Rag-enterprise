# Handoff → repo, CI and release hygiene

Written 2026-08-09 at the end of a long session. This document is for a fresh
session picking up the **repo/CI/GitHub side** and assumes no memory of the
previous ones.

Read `handoffTo6.md` first if you need the system itself — what it does, how it
is measured, what is deployed. That document is current as of this commit and
is the one to trust about the RAG pipeline. This one covers what changed today,
what state git and CI are in, and what is left.

---

## 1. Where everything stands in one paragraph

The project is a RAG system over real SEC 10-K filings with investment-bank
RBAC. All six phases of "Project 6" are built. **As of today all 38 spec
requirements are met** except the demo video, which the owner dropped
deliberately. `main` is clean, pushed, and **CI is green for the first time in
a while** — which matters more than it sounds, see §3. Two Cloud Run services
are live and serving the corpus the published numbers were measured on.

---

## 2. What today's session did

Seven commits, oldest first. Read the commit bodies — they carry the reasoning,
and several record mistakes worth not repeating.

| Commit | What |
|---|---|
| `f27257b` | Handoff rewrite; recorded that the demo video was dropped |
| `eb39a18` | Corrected an overstated claim about hybrid search |
| `83c6b04` | Made the RRF fusion weighting configurable (Phase 2.3) |
| `abc29ce` | Added `answer_correctness`; fingerprint the fusion weighting in eval runs |
| `3f1be15` | Measured all three chunking strategies — **the shipped one loses** |
| `66f48ab` | Let the image build without a prebuilt index again |
| `f207985` | Stopped a unit test depending on an API key |

Earlier in the same session, before `f27257b`: the deployment moved to the
measured index, and two privilege-escalation holes were closed. Those are
described in `handoffTo6.md` §2.1 and §2.2a.

### 2.1 The result that matters

~~`fixed` beat the shipped default by 19×, 22× and 8× the noise floor.~~
**Re-measured 2026-08-10 and cut to 2.8×, 2.3× and 3.0×.** The second run per
strategy happened; this is what it found.

The deltas held almost exactly (+0.093, +0.082, +0.130). The *floors* were the
problem: they came from one pair of `recursive` runs and were transferred to
the other strategies on an explicitly unverified assumption. Measured per
strategy they are 5–9× wider — faithfulness varies 0.034 on `semantic` against
0.007 on `recursive`. Two deltas that table counted, refusal correctness and
answer correctness, are now noise outright.

`recursive` stays the default. The trade is 0.045 of citation accuracy — the
strongest number in the project — for 0.08–0.13 elsewhere, and at 2–3× the
noise that does not carry enough to invalidate the shipped index, its digest,
the deployment and every published figure. `CASE_STUDY.md` has the re-scoring.

**The lesson worth keeping:** a floor measured once is a claim, not a
constant. Every multiple in that table was computed against a denominator with
n=1, and the denominator was where all the error was.

---

## 3. Git and CI state — the part this handoff is named for

- Branch `main`, at `f207985`, **clean and level with `origin/main`** (0 ahead,
  0 behind). Remote is `https://github.com/Murali-Sai/Rag-enterprise.git`.
- **CI is green.** `lint=success test=success`. Verify with
  `gh run list --limit 1`.

### 3.1 CI was lying for a long time, and here is how

`.github/workflows/ci.yml` has two jobs: `lint` (ruff check + `ruff format
--check src/ tests/`) and `test` (`pytest tests/unit`), and **`test` depends on
`lint`**. Lint had been failing on formatting — whitespace in three test files
— for at least five commits. While it failed, the test job reported **skipped,
not failed**, so the red badge said "formatting" and the suite was not running
at all.

Fixing the formatting let the suite execute, and it immediately failed on a
real defect: `test_a_pinned_mode_builds_a_fresh_pipeline` constructed a real
embedding client and needed `OPENAI_API_KEY`. It passed locally, where `.env`
supplies one, and failed anywhere that does not.

**The lesson to carry:** a red badge nobody reads hides the failures that
matter. If lint goes red again, the tests stop running, silently.

Reproduce the keyless condition before trusting a green local run:

```bash
OPENAI_API_KEY= ./.venv/Scripts/python.exe -m pytest tests/unit -q
```

371 pass under it today.

### 3.2 Open CI work — the likely subject of the next session

1. **`tests/integration` never runs in CI.** The workflow runs `pytest
   tests/unit` only, so 12 integration tests (`test_web_routes.py`) are
   untested on every push. They pass locally and take ~160s, which is probably
   why — but the exclusion is undocumented, so it reads as an oversight.
2. **Deprecated action runners.** `actions/checkout@v4`,
   `actions/setup-python@v5` and `codecov/codecov-action@v4` target Node 20,
   which GitHub has deprecated; they are being force-run on Node 24 with a
   warning on every job. Bump them.
3. **Codecov uploads without a token** and warns. Either add the token as a
   repository secret or drop the step.
4. **No badge in the README.** Now that CI means something, a status badge is
   worth adding — it was actively misleading before, so this had to come second.
5. **No branch protection.** Everything today went straight to `main`. If the
   repo is going in front of recruiters, a protected `main` plus PRs is a
   better story than 40 direct pushes, though it is cosmetic.

### 3.3 Repo hygiene worth a look

- `.gitignore` correctly excludes `chroma_data/` and `chroma_dist/`. Verified:
  `git ls-files | grep chroma_dist` returns nothing.
- `Project 6.docx` (the spec) is in the repo root and deliberately **not** in
  version control. A fresh clone will not have it. Read it with:
  `python -c "import zipfile,re,html; ..."` — pandoc is not installed here.
- `evaluation/results/` holds 20+ run files and is in git. That is deliberate:
  the corpus fingerprint in each is what makes historical claims checkable.
- Several stale log files sit in the root (`evaluation_run*.log`,
  `evaluation_baseline.log`, `evaluation_topk10.log`). Harmless, untidy.

---

## 4. Local state that is not in git

**This matters — a fresh clone cannot reproduce it, and it is expensive.**

`chroma_data/` (working index, ~470 MB now) holds four collections:

| Collection | Chunks | What |
|---|---|---|
| `rag_enterprise` | 8,232 | **The measured corpus**, digest `c2f8c13673cf5ca5`. Everything published describes this |
| `rag_enterprise_fixed` | 6,585 | Built today for the chunking comparison |
| `rag_enterprise_semantic` | 8,936 | Older, from the semantic-chunking experiment |
| `dedup_smoke_test` | 0 | Junk, safe to delete |

`chroma_dist/` (106 MB) is the pruned single-collection copy the image ships.
Regenerate by deleting it and running `make index-dist` — the script refuses to
finish unless the result is 8,232 chunks at that digest.

**Do not delete `chroma_data/`.** Rebuilding `rag_enterprise` costs an ingest
plus re-embedding, and the digest would change, invalidating every published
figure. If it is ever lost, say so in the docs rather than quietly re-ingesting.

---

## 5. Gotchas that will bite

**Use `./.venv/Scripts/python.exe`, never bare `python`.** The system Python is
3.13 with *some* dependencies but not `rank_bm25` or `sentence-transformers`.
The partial overlap is the trap: LangChain-only scripts run fine, then pytest
fails collection on five modules and reads as a broken suite.

**Two Docker traps, both of which made a test pass while measuring nothing.**
Both cost real time today:

- BuildKit reused the cached COPY layer when `chroma_dist/` was moved aside, so
  a "fresh clone" image still carried all 106 MB of the index and reported
  success. **Use `--no-cache` when testing the empty-index path.**
- The optional index COPY globs on `chroma_dist*`, which then matched
  `chroma_dist_held` — the directory the test itself had renamed it to. Move it
  **outside the build context** to test, not to a sibling. `.dockerignore` now
  excludes `chroma_dist_*/` so this cannot bite in a real build.

**`settings` is a singleton built at first import.** Writing to `os.environ`
afterwards reaches subprocesses and `os.execvp` but *not* the already-built
object. This produced a boot line reporting `text-embedding-3-small` over an
index just built with MiniLM — the exact misreport that line exists to prevent.
`scripts/start.py` now reads the env directly; anything else doing late
overrides needs the same care.

**~~The rate limit does not exist.~~ Fixed 2026-08-10** — it is enforced now,
on `/query` (20/min) and the two auth routes (30/min), keyed on the first
`X-Forwarded-For` hop. It was never the three-line change this said it was:
slowapi requires the endpoint to take parameters named exactly `request` and
`response`, and all three routes had named their Pydantic body `request`.

Carry the lesson rather than the fact. The bug that survived the first round of
tests was `headers_enabled=True` raising when an endpoint has no `response`
parameter — which only happens on the **success** path, because a wrong
password raises before the injection point. Every rejection test passed while
every successful login returned 500. Rate-limit suites test refusal by
instinct; the half that matters to a visitor is the other one, and only a live
request found it.

**`compare_eval_runs.py` calls identical `config` blocks a noise floor.** It
fingerprints *settings*, not source. Change behaviour without changing a
setting and it will call a real regression noise. Say so in the commit.

---

## 6. Cost and budget

Spend is on the owner's OpenAI key. **The owner's stated ceiling is under
$10/month to keep the project running**, and their instruction was no spend
after deployment from public traffic.

Today's session spent roughly **$2.50** — three eval runs (~$0.80 each) plus a
few cents of embeddings and manual testing. It was approved in advance.

Steady-state hosting, measured: **under $1/month.** Artifact Registry ~$0.40
(4.5 GB of images), Secret Manager ~$0.24 (4 active versions), Cloud Run $0
idle because both services scale to zero. The OpenAI key on the service is for
*embeddings only* at ~$0.000005/query.

**What would break the $10 ceiling**, in order of likelihood: setting
`min-instances` above 0 (~$50/month — the classic Cloud Run bill shock);
accumulating deploy images (~$1/month per dozen); sustained abuse (~$4/day,
mostly Cloud Run compute rather than OpenAI). A **GCP billing budget alert at
$10** was suggested and not set up — it is the one safety net that watches every
meter at once, and it is free.

**Confirm before spending.** Every eval run is ~$0.80.

---

## 7. Open work, ranked

1. ~~The chunking result needs a second run per strategy.~~ **Done
   2026-08-10.** It cut the claim from 19×/22×/8× to 2.8×/2.3×/3.0× and the
   default stayed put. See §2.1.
2. ~~CI items in §3.2~~ — **done**, except branch protection. Both suites run,
   actions bumped to v7, codecov gated on a token, badge added.
3. **The mojibake ships.** Sample documents carry mangled em dashes from a
   Windows-encoded ingest, visible in the first chunk of the headline demo
   (`ACME FINANCIAL HOLDINGS â€… TRADING DESK PROCEDURES`). Fixing it needs a
   re-ingest *plus* a fresh eval *plus* republishing every figure, because the
   digest is printed on the landing page. See `handoffTo6.md` §2.3.
4. **The reranker silently refuses answerable questions** — still the biggest
   live defect. Do not fix it by lowering the threshold.
5. **Goldman Sachs context recall is a hard 0.000** with its leading hypothesis
   measured and refuted.
6. Optional: enforce the rate limit, cap upload size, re-run the six-config
   ablation (~$3, only meaningful as a complete set).

---

## 8. Decisions already made — do not relitigate

- **The demo video is dropped.** Nothing public promised one. The three-shot
  demo path in `handoffTo6.md` §9 stays because it is verified behaviour.
- **The deployment generates with Gemini, not `gpt-4o`.** A public
  unauthenticated URL running gpt-4o is ~$0.012/query with no enforced rate
  limit. Retrieval is the measured pipeline; generation is not. The README and
  landing page say exactly that — do not upgrade it to "the demo is the
  measured system".
- **The image ships the measured index, and the index COPY stays optional.**
  Shipping it briefly broke `make docker-up` on a fresh clone, which is the one
  thing Project 6 §5.3 asks the container to do. Both properties have to hold.
- **`/documents/ingest` is disabled in the image** (`ALLOW_RUNTIME_INGEST=false`).
  The demo's admin password is published on the landing page, so admin is a
  public role and must not be able to write to a fingerprinted corpus.
- **Self-registration is capped at `viewer`.**
- **RRF weights default to 1.0/1.0**, not the 0.7/0.3 the spec suggests,
  because the hybrid comparison was measured at equal weighting.
- **The case study does not claim hybrid beats dense-only**, because the
  measurement says otherwise. If someone asks for that reframing, it is the one
  thing in this repo worth pushing back on — the negative results and the
  retracted claims are what make the rest of the numbers credible.

---

## 9. Verifying it all still works

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q
./.venv/Scripts/python.exe -m ruff check src tests dashboard evaluation scripts
./.venv/Scripts/python.exe -m ruff format --check src tests
```

383 tests, ruff clean, format clean. The last of those is the one CI trips on.

Live services (both scale to zero, so the first request is slow):

- API + landing page — https://rag-enterprise-laa65asupq-uc.a.run.app
- Dashboard — https://rag-enterprise-dashboard-laa65asupq-uc.a.run.app

Deployed revision `rag-enterprise-00007-zpc`. Its boot log names the corpus it
is serving, which is the fastest way to confirm a deployment is what it claims:

```
Corpus: rag_enterprise — 8232 chunks, embedded with openai/text-embedding-3-small
```
