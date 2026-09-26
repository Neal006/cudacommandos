# Edge Cases: what to test on train and on test

Each case has an expected behaviour and where it's enforced: **UT** = unit test
(`tests/`), **M** = mlguard, **V** = `validate_submission.py`, **E** = experiment slice
report. Examples are real rows from the data unless marked (synthetic).

## D. Data / I/O

| # | Case | Expected | Check |
|---|---|---|---|
| D1 | Partial download (Priyanshu saw 253k S1 rows instead of 2.2M) | Stop before any work | L1 contract row counts (ANALYSIS §1.2) · UT |
| D2 | TSV read without `sep="\t"` | Impossible: one reader function only | UT |
| D3 | Embedded CSV quotes `""NIAGARA"" BUILDING` (6 train S2 rows) | Row count matches `wc -l − 1`; pandas and polars agree | UT on those 6 ids |
| D4 | CRLF line endings / BOM / trailing tab | Stripped on read; never written | UT |
| D5 | Non-UTF-8 bytes | `utf8-lossy`, logged count; must be 0 | L1 |
| D6 | Null vs empty address (~3% S2/S3) | Both → `addr_empty=True`; features defined (no NaN reaching the model) | UT · M (NaN loss) |
| D7 | Empty name | 0% observed; the code path must still not crash | UT (synthetic) |
| D8 | Duplicate entity_id in a file | 0 observed; contract fails if > 0 | L1 |
| D9 | S2/S3 numeric id collision (26,801) | Ids always keep their prefix | UT |
| D10 | Unseen country label (anything beyond US/India/France) | Generic rule table; every S1 still gets a row | UT (country="Germany") · M missing_rows |
| D11 | Country string variants (`india`, ` US`) | Compared after strip+casefold, never enumerated | UT |
| D12 | Out-of-memory mid-blocking | Chunk results persisted; resume from the last chunk | E01 |

## T. Text normalization

| # | Case | Example | Expected |
|---|---|---|---|
| T1 | Native Indic script name | `குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்` | Translit + skeleton → `klpl psns` = S1 `Global Business` |
| T2 | Native-script state | `महाराष्ट्र`, `தமிழ்நாடு`, `ગુજરાત`, `കേരളം` | → canonical `maharashtra` … |
| T3 | Mixed script in one field | `NO. C-21 … AMBATTUR, தமிழ்நாடு` | Per-run transliteration |
| T4 | ZWJ/ZWNJ inside Indic conjuncts | (synthetic) | Kept for translit, removed after |
| T5 | Injected accents in US/India text | `Léarning`, `Prívate`, `Fóod` | Accent fold → equal |
| T6 | Real French accents | `Président`, `Thénard`, `HÊTRES` | Folded both sides |
| T7 | Domain as name | `deltatelecommunication.com`, `heassociates.com`, `léo.com` | Stem segmented; acronym/initials kept |
| T8 | Handle as name | `@nizarmarine` | `@` stripped; stem segmented |
| T9 | Junk prefix | `-- Holloway Peak`, `<< Team Ecole` | Stripped |
| T10 | Legal suffix moved/duplicated | `LLC Moncada`, `PVT. ASTOR TRADING LTD.`, `private foundation excel agency limited` | `legal_form` extracted from any position |
| T11 | Brackets | `QHC Culture [EURL]`, `Real Modern Food (Limited)` | Content kept, brackets dropped |
| T12 | `&` / `and` / `et` | `Lee & Lwson`, `Thermal & Fils` | Unified |
| T13 | Possessives | `Orelee's` / `Orelees` | Equal |
| T14 | Token order | `EVE'S ASSOCIATES HIGHLAND` | Token-set features |
| T15 | Extra noise word | `Lee and Lawson LLC Center` | `noise_word_extra` feature, not a hard mismatch |
| T16 | Name that is only a legal suffix / only generic tokens | `Primary Care Group` | `name_core` empty → fall back to `name_norm`; genericity high |
| T17 | Numbers inside names | `X 7 N`, `B+ Retail`, `N+ 0tg` (zero for O) | `name_num_agree`; `0`↔`o` fold only in the skeleton |
| T18 | Single-letter names / initials | `U & I`, `K/D Cadenza` | Not dropped by the length filter |
| T19 | French abbreviations | `R. DE DIEPPE`, `AV LEON JOUHAUX`, `ALLÉE`, `9 B CITÉ` | FR table: rue, avenue, allee, bis |
| T20 | French admin level mismatch | S1 `Hauts-de-France` vs S2 `Nord` | Department→region table |
| T21 | US state abbrev ↔ full | `NY` ↔ `New York`, `Ohio, 991 …` | State canon |
| T22 | India abbrev | `MH` ↔ `Maharashtra`, `RD`/`MG Rd` | Tables |

## A. Address semantics

| # | Case | Example | Expected |
|---|---|---|---|
| A1 | House number typos | `9914` vs `914` / `991` | `hn_fuzzy`=1, `hn_exact`=0 |
| A2 | Fractions / suffixes | `599 1/2`, `5 bis`, `17-A` | Parsed as the same base number |
| A3 | Unit/PMB/PO box noise | `PMB 9531`, `PO Box 3545`, `Unit LOT 130` | Moved to `addr_extra` |
| A4 | Different locality level | `POTTERSVILLE` vs `Horicon` | locality overlap low, street + number decide |
| A5 | Landmark-based | `Nr. Jhansi Ki Rani Statue`, `Near SBI ATM` | Stop words in core; not a mismatch |
| A6 | Component reorder | `IA, Iowa City, 1064 Newton Rd` | Order-free locality/street features |
| A7 | Indian numbering | `KH NO. -570/13`, `G-3/571`, `104/1/1` | Slash groups kept as one token |
| A8 | Empty address on one side | 3% S2/S3 | Name-only path; `addr_empty_any` |
| A9 | No number on one side | 12% of positives | `hn_*` = missing (not 0) |

## B. Blocking

| # | Case | Expected | Check |
|---|---|---|---|
| B1 | S1 with zero candidates (orphan) | Still gets an empty row in both files | M missing_rows · existing `orphan_check` |
| B2 | All tokens pruned by `max_df` (generic name + generic address) | Pass B/C still produce candidates | E03 slice |
| B3 | Candidate duplicated across passes | Union dedups; one row per (s1, cand) | UT |
| B4 | Ties at rank K | Deterministic tie-break (by id) | UT |
| B5 | A match that crosses countries | Diagnostic only (0 on train) | E04 variant |
| B6 | Very dense India posting lists | Chunk memory bounded | E01 |

## M. Model and decision

| # | Case | Expected | Check |
|---|---|---|---|
| M1 | True singleton with a strong look-alike (same name, same locality) | Empty prediction when house number/street disagree | E slice "hard negatives" |
| M2 | Entity with 8–10+ true matches | Not truncated by the decision (no cap below the K union) | E slice by true count |
| M3 | One S2/S3 claimed by two S1s | Goes to the argmax only | M partition · UT |
| M4 | Co-located businesses (117k S1 share an address) | Name decides; `colocation` feature | E slice |
| M5 | Generic names (`primary care group` ×253) | Address decides; `name_genericity` | E slice |
| M6 | Native-script candidate | translit/skeleton features carry it | E slice |
| M7 | France | Predicted singleton rate and links/entity ≈ OOF | M test_singleton_drift / test_links_drift |
| M8 | Calibration outside the training range | Isotonic clipped | UT |
| M9 | NaN / inf features | Filled with a documented default; never reach LightGBM silently | UT · M nan_loss |
| M10 | Feature column order differs train vs test | `reindex(columns=model.feature_name())` (exists) plus an assert that no column was *filled* | UT |

## O. Output

| # | Case | Expected | Check |
|---|---|---|---|
| O1 | Exactly 1,732,544 rows, one per test S1 | ✓ | V · M |
| O2 | Empty list written as an empty field (`S1-x\t`) | ✓ | V |
| O3 | No spaces after commas, no quotes | ✓ | V · M |
| O4 | Matches ⊆ candidates | ✓ | V (warn) · M (fail) |
| O5 | `candidate_pairs.tsv` is the scored set, not an earlier stage | ✓ | Code review + the row sum equals the scored-pair count |
| O6 | Windows writes `\r\n` | `newline="\n"` | UT |
| O7 | File > upload limit? (~100 MB) | Check the portal limit early | manual |

## R. Run / operations

| # | Case | Expected |
|---|---|---|
| R1 | Pipeline crashes mid-run | `train_guarded.sh` still writes the end event; the watcher exits; cache intact |
| R2 | Stale cache after a normalizer change | Cache key includes the normalizer version hash |
| R3 | Seeds | Fixed (`C.SEED`) in sampling, folds, LightGBM; rerun reproduces OOF to 1e-4 |
| R4 | Different machine (AWS vs laptop) | Same outputs (hash) from the same cache |
| R5 | Last-hour upload failure | Reserve slot #5 on day 3; validated files kept per tag |
