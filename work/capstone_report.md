# Capstone Report — Refresh / Content Opportunity Scoring

* **Author:** Ahmed Mahmoud
* **Lane:** Refresh / Content Opportunity Scoring
* **Repo:** https://github.com/AhmedMahmoud-123/FlyRank_AI
* **Date:** August 2026

## 1. Problem framing

This project supports the decision of **which content items a FlyRank editor should review first for possible refresh or improvement**.

The unit of analysis is a content item for a client. The output is a ranked action queue containing an action score and a plain-language reason code. A human editor can use the queue to prioritize pages for review rather than reviewing all pages equally.

The cost of a wrong call is asymmetric: a low-value page may receive unnecessary editorial attention, while an important page may be missed. For that reason, the system is designed as **decision support rather than automatic content editing**.

Data/ML is useful because the dataset contains many content items with different levels of impressions, search position, freshness, and query-level behavior. A ranking model can combine these signals and surface items that a simple rule may not prioritize.

## 2. Data safety

The analysis uses the FlyRank search dataset prepared through the feature-building workflow. The final modeling table contains **93,401 content items**, with **68,798 training rows and 24,603 test rows**. The test set contains **11 clients** and has an observed declining-label base rate of **73.9%**.

The final feature set was:

* `impressions_prev30`
* `avg_position_prev30`
* `days_since_last_update`
* `query_impressions_prev30`
* `query_concentration_prev30`

The following fields were deliberately excluded from the final model:

* `trend_direction` — label-derived and therefore a leakage risk.
* `trend_pct` — label-derived / outcome-derived information and therefore excluded.
* `visible_queries`
* `rare_share`
* `anon_share` — treated as suspect query-level fields because their relationship to the target could contain information unavailable or inappropriate for a clean prediction setting.
* `client_hash_id` — used for grouping and train/test separation, never as a model feature.
* `content_hash_id` — used as an identifier, never as a model feature.
* `is_declining_label` — used only as the target.
* `is_test_client` — used only to define the evaluation split.

The final feature-building workflow produced **133,852 content items with aligned query-level features** before the final content-level merge. The merge contained **0 duplicate `(client_hash_id, content_hash_id)` pairs** and preserved the expected **93,401 rows**.

Query-level features matched **92.6%** of the final rows. There were **8,787 rows with no usable query-impression signal**. Of the rows with query features, **84,614** had all query features present. There were **1,915 rows with zero query impressions** and **8,787 rows with missing query concentration**. No row had positive query impressions while having a missing concentration value.

No client names, domains, private queries, credentials, or other client-identifying information are used in the report or modeling features.

## 3. Baseline

The baseline was a transparent rule designed around the same refresh-review logic used earlier in the project.

A page receives a positive baseline action score when it has at least **500 impressions in the previous 30 days** and is either:

* at least 180 days since its last update, or
* worse than the observed median positive average position.

The score is then weighted by `impressions_prev30`, so the baseline naturally prioritizes high-visibility pages that also meet the review conditions.

On the held-out test clients:

| Metric        | Baseline | Final model | Test base rate |
| ------------- | -------: | ----------: | -------------: |
| Precision@20  |    1.000 |       0.900 |          0.739 |
| Precision@50  |    1.000 |       0.860 |          0.739 |
| Precision@100 |    0.950 |       0.930 |          0.739 |
| Precision@200 |    0.915 |       0.900 |          0.739 |

The baseline therefore performs very strongly at the highest ranks in this dataset. The model does not beat the baseline on Precision@K, so the result should **not** be presented as evidence that the model is more accurate than the rule.

The baseline's Precision@20 is **1.000**, compared with a test base rate of **0.739**. This represents a **+0.261 absolute lift over the base rate**, or approximately **+35.4% relative lift**.

The final model's Precision@20 is **0.900**, representing a **+0.161 absolute lift over the base rate**, or approximately **+21.8% relative lift**.

The baseline and model nevertheless surface substantially different content. Their top-500 lists have **0 items in common (0.0% overlap)**.

The baseline top-500 pages have a mean of approximately **24,441 impressions/30d**, compared with approximately **2,169 impressions/30d** for the model top-500 pages.

This means the baseline behaves largely like a high-visibility declining-page queue, while the model identifies many lower-traffic pages that the baseline does not prioritize.

## 4. Model / analysis

The final model is a `HistGradientBoostingClassifier`.

It was selected because the task is binary classification with a mixture of continuous search-performance and freshness features, and the model can capture nonlinear relationships between these signals.

The target is:

> `is_declining_label`, representing the observed declining-content label supplied by the prepared dataset.

The final leakage-clean feature set contains:

```text
impressions_prev30
avg_position_prev30
days_since_last_update
query_impressions_prev30
query_concentration_prev30
```

Several model specifications were compared:

* **Baseline features only:** AUC = 0.592
* **Baseline + all query features:** AUC = 0.675
* **Baseline + safe query features:** AUC = 0.615
* **Baseline + suspect query features:** AUC = 0.650
* **Final leakage-clean model:** AUC = 0.615

The suspect-feature experiment was retained as a diagnostic rather than used for the final model. The final model uses only the baseline signals plus the two query-level features judged appropriate for the leakage-clean feature set.

The results show that the query-enhanced model specifications can improve discrimination relative to the baseline-only feature set, but the final leakage-clean model has only modest discrimination with **AUC = 0.615**.

## 5. Evaluation

The evaluation uses a **client-level holdout split**. Training contains **68,798 rows**, while the held-out test set contains **24,603 rows from 11 clients**.

The split is important because content from the same client can share structural patterns. Keeping test clients separate provides a more demanding evaluation than randomly mixing content items from the same clients between training and testing.

The observed test base rate is **73.9%**. This is reported alongside Precision@K because precision values should be interpreted relative to the underlying prevalence of the positive label.

The final model achieved:

* **AUC = 0.615**
* **Precision@20 = 0.900**
* **Precision@50 = 0.860**
* **Precision@100 = 0.930**
* **Precision@200 = 0.900**

The baseline achieved:

* **Precision@20 = 1.000**
* **Precision@50 = 1.000**
* **Precision@100 = 0.950**
* **Precision@200 = 0.915**

The model therefore does not replace the baseline on top-k precision in this experiment.

The model score distribution across all **93,401 content items** was:

| Statistic | Model score |
| --------- | ----------: |
| Count     |      93,401 |
| Mean      |       0.655 |
| Std       |       0.178 |
| Minimum   |       0.046 |
| 25%       |       0.566 |
| Median    |       0.693 |
| 75%       |       0.779 |
| Maximum   |       0.976 |

Using a 0.5 model-score threshold, the baseline rule flagged **18,231 items (19.5%)**, while the model flagged **76,074 items (81.4%)**.

A diagnostic comparison identified **62,571 items where the rule and model disagreed**. This disagreement set is useful for human review because it represents content where the transparent rule and learned model produce different screening decisions.

The model threshold of 0.5 should not be interpreted as a validated production threshold. It was used for diagnostic comparison between the model and rule.

The top-500 ranking comparison showed:

```text
Baseline top-500: 500
Model top-500:    500
Top-500 overlap:  0 items (0.0%)
```

The traffic distributions were also substantially different:

```text
Baseline top-500 mean impressions/30d: 24,441
Model top-500 mean impressions/30d:     2,169
```

For the baseline top-500:

| Statistic | impressions_prev30 |
| --------- | -----------------: |
| Count     |                500 |
| Mean      |         24,440.682 |
| Std       |         18,974.914 |
| Minimum   |             12,104 |
| 25%       |         14,511.750 |
| Median    |             18,107 |
| 75%       |         27,252.500 |
| Maximum   |            183,954 |

For the model top-500:

| Statistic | impressions_prev30 |
| --------- | -----------------: |
| Count     |                500 |
| Mean      |          2,169.074 |
| Std       |          7,958.319 |
| Minimum   |                100 |
| 25%       |            253.500 |
| Median    |                761 |
| 75%       |          1,026.250 |
| Maximum   |            112,236 |

This provides evidence that the model is not simply reproducing the baseline's high-traffic ranking.

## 6. Interpretation

The model found that query-level behavior adds directional information beyond the basic refresh signals.

Permutation importance on the held-out test data ranked the final features approximately as follows:

| Feature                      | Permutation importance |
| ---------------------------- | ---------------------: |
| `impressions_prev30`         |                0.04508 |
| `query_concentration_prev30` |                0.04219 |
| `days_since_last_update`     |                0.01392 |
| `query_impressions_prev30`   |                0.01120 |
| `avg_position_prev30`        |               -0.00132 |

The strongest measured signals were therefore **recent impressions and query concentration**, followed by freshness and query impressions. Average position contributed essentially no positive permutation importance in this particular analysis.

The negative value for `avg_position_prev30` should not be interpreted as evidence that average position is causally harmful. It means that, under this particular permutation-importance calculation, shuffling that feature did not reduce test performance and produced a very small improvement instead.

The partial-dependence analysis for `impressions_prev30` showed that predicted declining probability generally increased from very low impression levels into the lower-thousands range and then flattened rather than continuing to rise strongly.

Selected model predictions were:

```text
impressions_prev30 =        118   predicted P(declining) = 0.529
impressions_prev30 =        545   predicted P(declining) = 0.699
impressions_prev30 =        973   predicted P(declining) = 0.733
impressions_prev30 =       1400   predicted P(declining) = 0.755
impressions_prev30 =       1828   predicted P(declining) = 0.753
impressions_prev30 =       2255   predicted P(declining) = 0.750
impressions_prev30 =       2682   predicted P(declining) = 0.752
impressions_prev30 =       3110   predicted P(declining) = 0.750
impressions_prev30 =       3537   predicted P(declining) = 0.737
impressions_prev30 =       3965   predicted P(declining) = 0.730
impressions_prev30 =       4392   predicted P(declining) = 0.729
impressions_prev30 =       4819   predicted P(declining) = 0.729
impressions_prev30 =       5247   predicted P(declining) = 0.744
impressions_prev30 =       5674   predicted P(declining) = 0.744
impressions_prev30 =       6102   predicted P(declining) = 0.744
impressions_prev30 =       6529   predicted P(declining) = 0.729
impressions_prev30 =       6956   predicted P(declining) = 0.729
impressions_prev30 =       7384   predicted P(declining) = 0.748
impressions_prev30 =       7811   predicted P(declining) = 0.750
impressions_prev30 =       8239   predicted P(declining) = 0.751
```

For example, the predicted probability increases from approximately **0.529 at 118 impressions** to **0.699 at 545 impressions**, reaches approximately **0.755 around 1,400 impressions**, and then largely flattens around the 0.73–0.75 range.

This should be interpreted as an observed model relationship, not a causal effect of impressions on decline.

A major negative result is that the trained model does **not** outperform the transparent baseline on Precision@K. The useful finding is instead that the model produces a substantially different ranking.

The top-500 overlap is **0 items out of 500 (0.0%)**, meaning the two approaches select completely different top-ranked populations in this rerun.

The model top-500 also has a much lower mean traffic level than the baseline top-500:

```text
Baseline: 24,441 impressions/30d
Model:     2,169 impressions/30d
```

This supports the interpretation that the model is surfacing smaller pages that the visibility-based baseline largely ignores.

These are observed and directional relationships in this dataset. They do not establish causality or prove how a search engine ranks content.

## 7. Recommendation

The output should be used as a **ranked human-review playbook**, not an automated editing system.

### Priority 1 — Review high-scoring model items

Use the model score to identify content that may deserve review even when it does not satisfy the baseline rule. This is particularly useful for lower-traffic pages that would otherwise be absent from the baseline queue.

The model's top-500 ranking is substantially different from the baseline, with **0% top-500 overlap**. This makes the model potentially useful as a complementary discovery mechanism rather than as a replacement for the baseline.

### Priority 2 — Review high-visibility baseline items

Continue using the transparent baseline for pages with substantial impressions that are stale or have weak observed position. These pages have strong practical visibility and should remain an important review population.

The baseline top-500 has a mean of **24,441 impressions/30d**, substantially higher than the model top-500 mean of **2,169 impressions/30d**.

### Priority 3 — Investigate model/rule disagreements

The **62,571-item disagreement set** should be treated as a diagnostic review population.

Items where the model assigns a high score but the baseline assigns no action are particularly useful for understanding what additional query-level patterns the model is capturing.

Because the model flags **81.4%** of all content at the 0.5 threshold while the rule flags only **19.5%**, the 0.5 threshold is clearly not appropriate as an assumed production action threshold without further calibration and business-capacity analysis.

### Priority 4 — Use reason codes as review prompts

Reason codes should remain simple review explanations:

* `stale_but_visible` — high recent visibility and an update age of at least 180 days.
* `position_slipping` — average position is worse than the observed median.
* `standard_review` — does not satisfy either stronger rule condition.

These codes are **not causal diagnoses**. They explain the screening logic and should be verified by an editor.

### Confidence and limits

Confidence is moderate for using the output as a **screening and prioritization aid** within this dataset. Confidence is lower for claiming that the model will improve future traffic, rankings, or refresh outcomes.

The baseline currently has better measured Precision@K on the held-out test clients. The strongest practical case for the model is therefore not that it replaces the baseline, but that it provides a **different candidate pool**, especially among lower-traffic content.

The final model's AUC of **0.615** indicates only modest discrimination, so the score should not be treated as a highly accurate probability of future decline.

The very high model flag rate at a 0.5 threshold also shows that the model score should be used primarily for **ranking**, rather than treating 0.5 as an automatically meaningful yes/no cutoff.

No page should be automatically edited, published, merged, pruned, or promised an outcome because of the score alone.

## 8. Reproducibility

The project is maintained in the public repository:

`https://github.com/AhmedMahmoud-123/FlyRank_AI`

The capstone notebook is located at:

`work/notebooks/capstone.ipynb`

The capstone report is intended to be saved as:

`work/capstone_report.md`

The analysis uses fixed random seeds where applicable, including:

```python
random_state=42
```

The main modeling workflow uses scikit-learn's `HistGradientBoostingClassifier`, with the final evaluation performed against a client-level holdout.

The final dataset contains:

```text
93,401 eligible content items
68,798 training rows
24,603 test rows
11 held-out test clients
73.9% test positive rate
```

The query-level feature preparation produced:

```text
133,852 aligned query-feature content items
0 duplicate client/content pairs
92.6% query-feature coverage
8,787 rows with no usable query-impression signal
```

The exported research artifacts include:

```text
work/outputs/artifacts/precision_at_k.png
work/outputs/artifacts/partial_dependence_impressions.png
work/outputs/artifacts/rule_vs_model_traffic_dist.png
```

The final action queue and evaluation outputs are also written under `work/outputs/`.

The intended reproducibility workflow is to clone the repository, open `work/notebooks/capstone.ipynb`, and run the notebook from the prepared project environment. The notebook contains the feature preparation, query-level feature construction, leakage checks, baseline, model evaluation, ranked recommendations, interpretation, and paper artifacts.

The analysis is designed to be reproducible from the prepared FlyRank dataset without exposing client-identifying information.

---

## Claims checklist

The results in this report use **observed, measured, directional, and decision-support** language.

The test base rate of **73.9%** is reported alongside Precision@K.

The final model's measured **AUC is 0.615**, while the baseline is stronger on the reported Precision@K metrics.

The final model achieved:

```text
Precision@20  = 0.900
Precision@50  = 0.860
Precision@100 = 0.930
Precision@200 = 0.900
```

The baseline achieved:

```text
Precision@20  = 1.000
Precision@50  = 1.000
Precision@100 = 0.950
Precision@200 = 0.915
```

The baseline therefore remains stronger on the reported top-k precision metrics.

The model and baseline have **0 overlapping items in their top-500 lists (0.0%)**, showing that they prioritize substantially different content.

The baseline top-500 has a mean of **24,441 impressions/30d**, compared with **2,169 impressions/30d** for the model top-500.

The final model's strongest permutation-importance signals are `impressions_prev30` and `query_concentration_prev30`.

The model's partial-dependence analysis shows an increase in predicted declining probability from very low impression levels followed by a broad flattening pattern.

The model flags **76,074 of 93,401 items (81.4%)** at a 0.5 threshold, compared with **18,231 items (19.5%)** for the baseline rule. This threshold is treated as diagnostic rather than a validated production decision threshold.

The diagnostic rule/model disagreement set contains **62,571 items**.

No causal claims are made about content refreshes, traffic changes, or search-engine algorithms.

The analysis does not claim to predict Google's algorithm.

No client-identifying information is included.

The report's numerical claims are based on the latest completed capstone notebook rerun.
