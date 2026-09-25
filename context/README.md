# Context

Raw source material for the challenge. Everything here came from outside the
repo — organisers, AWS, the team chat, or a teammate's machine — and is kept
unedited so the derived documents in `docs/` can always be traced back.

**`context/` holds inputs. `docs/` holds conclusions.** If a number appears in
`docs/DATA_BRIEF.md`, it was measured from the dataset or taken from something
in here.

---

## challenge/

| File | Source | Why it matters |
|---|---|---|
| `problem_statement.pdf` | Unstop portal, 25 Sep | The authoritative spec: format, metric, constraints, fair-play rules |
| `guidelines.pdf` | Unstop portal, 25 Sep | Challenge window, **5 submissions/day**, artefacts required, login rules |
| `dataset_README.md` | Shipped inside the dataset package | Same content as the problem statement; kept because it is what ships beside the data |
| `Documentation_template_original.md` | Dataset package | Pristine copy. The working copy lives at the repo root and gets filled in — keep this one untouched for diffing |

## aws/

| File | Source | Why it matters |
|---|---|---|
| `aws_credits_instructions.pdf` | Organisers, pre-challenge | Builder ID, Free Tier, credit redemption, cross-account S3 relay |
| `team_bucket_announcement.md` | Team chat, 25 Sep | Bucket name, region, access model — plus why we authenticate differently |

## prep/

| File | Source | Why it matters |
|---|---|---|
| `prep-blog.txt` | AWS Builder Center prep blog | Timeline, prizes, eligibility, and the Builder Center free sandbox (8h/week, no card) |

The Twitch recording of the 21 Sep prep session is not archived here — Twitch
does not expose a transcript. The blog covers the same ground.

## teammates/

| File | Source | Why it matters |
|---|---|---|
| `krina_eda.ipynb` | Krina, 25 Sep | First EDA pass. **Read the caveat below before trusting its output.** |

### Caveat on `krina_eda.ipynb`

The notebook reports `S1 total: 253,300`, `S1 labeled: 18,234` and an overlap
of only `2,116` IDs — which would mean the ground truth barely matches the
source file at all.

That is an artifact of a **partially-extracted download**, not a property of
the data. On the complete files:

```
train_source1 entity_ids   2,206,821
ground_truth  entity_ids   2,206,821
present in both            2,206,821
missing either direction            0
```

Exactly one ground-truth row per Source-1 record. The notebook's other cells
(schema, sample rows, ID-length distribution) are still fine — only the
counts and the overlap conclusion are affected. Recorded in
`docs/DATA_BRIEF.md` §2 as well, so nobody rediscovers it.

**If your row counts disagree with `DATA_BRIEF.md` §1, re-extract first.**

---

## Not stored here

- **The dataset itself** (~2.4 GB) — lives outside the repo, see the README.
  `.gitignore` blocks `*.tsv` for this reason.
- **Screenshots** pasted into chat (AWS free-plan status, Explore AWS
  activities, the SageMaker domain-quota error). The facts from them are
  recorded in `docs/AWS_SETUP.md` and the memory notes; the images add nothing
  the text does not.
- **Anything with credentials in it.** Account IDs are fine and appear in the
  docs; access keys, tokens and session material never enter the repo.
