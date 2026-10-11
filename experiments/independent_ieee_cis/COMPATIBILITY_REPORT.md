# IEEE-CIS Fraud Detection — compatibility report (proposal stage)

Status: **assessment only. No data were downloaded, no competition terms were accepted, no account was created, nothing was run.** The owner chose IEEE-CIS as the primary public candidate;
this report says what it can and cannot support. Evidence tags used throughout:

* **[S]** — stated by a source read in this session. Limitation: the host `www.kaggle.com` (and `arxiv.org`, `ieee-dataport.org`, `towardsdatascience.com`) were **not reachable** from this environment (network policy: DNS refused),
  so the *official* Kaggle data page and rules were **not read**. [S] facts below come from search-result summaries of secondary pages that quote the description; they are not verified against the official source.
* **[M]** — from the author's background knowledge, **not verified here**. Must be checked on the actual files (stage 0 of the protocol) before anything depends on it.
* **[A]** — an assumption or design choice of this project.
* **[CODE]** — read in this repository.

## 1. Provenance and access

| Item | Content | Tag |
|---|---|---|
| Origin | Card-transaction data donated by Vesta Corporation for the IEEE Computational Intelligence Society's Kaggle challenge "IEEE-CIS Fraud Detection" (https://www.kaggle.com/c/ieee-fraud-detection) | [S] (secondary summaries) |
| Real or synthetic | Real e-commerce card transactions, anonymised and masked | [S]/[M] |
| Size | 590,540 transactions, 20,663 fraudulent (3.5 %) — one summary; other summaries quote 431 features (400 numeric, 31 categorical) and 394 columns for the transaction table, so the column counts disagree between secondary sources | [S], inconsistent |
| Files | Two joined tables, **transaction** and **identity**, keyed by `TransactionID`; not every transaction has an identity row | [S] |
| Official split | competition `train_*` (labelled) and `test_*` (labels **not published**; scored only through the competition's submission system) | [M] |
| Access | Download requires a Kaggle account and acceptance of the competition rules (one guide: downloading without joining returns HTTP 403) | [S] |
| Licence / permitted use | **Unknown.** The rules page was not read. Competition data usually carry use restrictions (competition use, no redistribution, sometimes limits on publication or commercial use). Must be read and accepted by the owner personally | [A] — **owner decision D2** |
| Network | This environment cannot reach Kaggle. The owner would have to download locally and provide the files, or add the host to the environment's allowed domains (Network access setting) | [CODE]/env |

## 2. Schema (what is stated, what must be inspected)

| Group | Content | Tag |
|---|---|---|
| Target | `isFraud` (0/1) in the training transaction table | [M] (name); [S] (labelled fraud share 3.5 %) |
| Time | `TransactionDT`: a *timedelta from a reference datetime, not an actual timestamp* | [S]. Units (seconds) and span (about six months in the training file) | [M] |
| Amount, product | `TransactionAmt`, `ProductCD` (a handful of categories) | [S] names; [M] cardinalities |
| Card / billing | `card1`–`card6`, `addr1`, `addr2`, `dist1`, `dist2`, `P_emaildomain`, `R_emaildomain` (masked or coarse) | [S] names |
| Vesta-engineered | `C1`–`C14` (counts), `D1`–`D15` (timedeltas), `M1`–`M9` (match flags), `V1`–`V339` (engineered); meanings are **masked** | [S] "masked, engineered"; family names [M] |
| Identity | `id_01`–`id_38`, `DeviceType`, `DeviceInfo` (network, browser, device information) for a subset of transactions | [M]; coverage roughly a quarter of rows [M] |

Everything in the [M] column is to be confirmed by a **label-blind schema audit** (protocol stage 0: column list, dtypes, missingness, ranges of `TransactionDT`, cardinality of identifiers; no label statistics).

## 3. Can the dataset support the project's objectives?

| Objective | Verdict | Reason |
|---|---|---|
| Transaction-level fraud classification | **Yes** | labelled rows, 3.5 % positives, mixed numeric/categorical fields |
| Chronological train / validation / test | **Yes, inside the labelled file only** | `TransactionDT` orders rows. The official test file has no public labels, so the final hold-out must be the *last chronological part of the labelled file* [M]. Right-censoring risk: see §5 |
| Meaningful transaction histories | **Only through a proxy entity** | there is **no customer or account id** [M]. A history exists only if rows can be grouped (§4). The LSTM needs ≥ 10 earlier rows per entity: the share of rows that qualify is unknown until the audit |
| Defensible behavioural grouping | **Uncertain — the central risk** | card fields are masked identifiers of a payment instrument, not a person; see §4 |
| Contribution of the LSTM | **Testable, conditionally** | if a proxy entity with enough history exists, the same ablation as Experiment 3 (forest with vs without the sequence score) can be run; if histories are short or noisy, a null result cannot be interpreted |
| First-fraud detection | **Conditionally** | needs an entity grouping; "first fraud of an entity" then means the first labelled fraud of the proxy entity. Label construction matters (§5) |
| Warning lead time | **No (not supported)** | no login/precursor events and no ground-truth warning window; labels are chargeback-based and arrive late. Any "alerts before the first fraud" are indistinguishable from false positives. Lead time must not be reported for this dataset |
| SHAP explanations | **Yes, for the model actually trained** | Tree SHAP on the forest/gradient-boosting model over the adapted features; explanations of masked Vesta features (`V*`, `C*`, `D*`) would not be human-readable |

## 4. Grouping keys and their limits (do not assume a card id is a customer)

| Candidate entity | Construction | Limits |
|---|---|---|
| E1 `card1` | one masked card field [S] | one value can be shared by unrelated cards (collisions) and one customer can hold several; no guarantee of uniqueness |
| E2 `card1`+`card2`…`card6` (+`addr1`) | combined card and billing fields | fewer collisions, but missing values split histories; an address change splits a customer |
| E3 community-style `uid` | card fields + `addr1` + an *account-start day* derived from `TransactionDT` and a Vesta timedelta (`D1`) | a heuristic invented by competitors [M], not documented by the organiser; `D1`'s meaning is masked — if it is not strictly past-only the entity itself could leak future information |
| E4 email domain, device | `P_emaildomain`, `DeviceInfo` | shared by many people (large providers); sparse |

Consequences: (i) results are **per proxy entity**, never "per customer"; (ii) the primary entity is chosen **label-blind** by pre-registered consistency diagnostics (within-entity agreement of `card4`, `card6`, `ProductCD`, address, email domain; history length) and the others are sensitivity analyses; (iii) cluster bootstrap must use the entity (and, if entities overlap, connected components); (iv) if no candidate gives a stable history, the sequence question is **not answerable** on this dataset and the report says so.

## 5. Labels, time and leakage risks

| Risk | Detail | Tag | Control |
|---|---|---|---|
| Label propagation | The organiser has described labelling as chargeback-based, with later transactions linked to a reported card/account/email/address also labelled fraud, and "legitimate" if nothing was reported within a window (about 120 days) | [M] — **verify in the organiser's data description before use** | If true, "subsequent fraud" rows are partly artefacts of linking; report first-fraud-per-entity separately from all-fraud; never present recall on propagated rows as early detection |
| Right-censoring | Frauds not yet reported at the end of the period look legitimate; late-period fraud rate is understated | [M]/[A] | purge/embargo between segments, report fraud rate per week of the *training* part only, treat the last weeks of the labelled file as unreliable for the hold-out (decision D5) |
| Vesta features | `C*`, `D*`, `V*`, `M*` are produced by the data owner; whether each is computed from past data only is unknown | [M] | excluded from the primary model; a separate "Vesta-feature" tabular arm is context only |
| Identifier leakage | `TransactionID` is believed to increase with time; card/entity ids can encode the label through the label-propagation rule | [M] | forbidden as inputs; ID-relabelling invariance test (Experiment 2's L14) |
| Entity built from future | e3 `uid` uses a timedelta of unknown semantics | [M] | only candidates computed from past rows of the same grouping are eligible for primary |
| Random split | rows of one entity in both train and test | — | chronological split and entity-grouped purge |
| Target encoding / scaling | fitted on all data | — | fit on training segment only |
| Identity-table join | identity rows exist for a subset; presence itself may correlate with fraud | [M] | include presence as an explicit, reported feature only in an ablation |
| Duplicates | repeated rows | [M] | check in stage 0 |
| Hold-out reuse | repeated looks at the last segment | — | single-use ledger as in Experiments 2/3 |

## 6. What cannot be claimed

Not the same customers, banking environment, feature definitions or fraud-generation process as the synthetic data. Not "the platform was validated on real data": at most, *an adapted model trained from scratch on a public dataset with a proxy entity,
evaluated chronologically, behaves as follows*. No statement about the saved models or the nine Behavioral Fraud DNA features as deployed. No early-warning claim. No claim about explanation quality beyond "SHAP values of the model that was trained".

## 7. Expected size and resources

590,540 labelled rows [S]; the transaction file is several hundred MB as CSV [M]; in memory as float32 with all columns about 1–2 GB [M]. Environment today: 4 CPU cores, 15 GB RAM, 15 GB free disk, no GPU.
Feature building with the project's per-row builder took 441 s per 416 k rows [Experiment 2 measurement]; a vectorised builder is planned. See `RESOURCE_AND_EXECUTION_PLAN.md`.

## 8. Alternatives if IEEE-CIS fails the audit

No dataset was verified. Candidates the literature uses (availability, licence and suitability **unchecked**): a simulated card-transaction set with customer and merchant ids (Sparkov-type, synthetic), a multi-institution synthetic transfer set (PaySim-type, synthetic, account ids but known simplifications),
a tabular-sequence transaction set with user and card ids (TabFormer-type, synthetic), and the 2013 European card set (no entity id; PCA-masked; not suitable for histories). Using any of them as "independent" evidence has the same limit as the synthetic data: it is still synthetic. Selection criteria are in `DATASET_SELECTION_CHECKLIST.md`.
