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

Query-level features were checked for missingness. Query features matched **92.6%** of the final rows. There were **8,787 rows with no usable query-impression signal**, consisting of missing query impressions or zero query impressions. No row had positive query impressions while having a missing query-concentration value.

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
| Precision@20  |    1.000 |       0.950 |          0.739 |
| Precision@50  |    1.000 |       0.940 |          0.739 |
| Precision@100 |    0.950 |       0.920 |          0.739 |
| Precision@200 |    0.915 |       0.910 |          0.739 |

The baseline therefore performs very strongly at the highest ranks in this dataset. The model does not beat the baseline on Precision@K, so the result should **not** be presented as evidence that the model is more accurate than the rule.

However, the model and baseline surface very different content. Their top-500 lists overlap by only **1 item (0.2%)**.

The baseline top-500 pages have a mean of approximately **24,441 impressions/30d**, compared with approximately **2,470 impressions/30d** for the model top-500 pages.

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

* **Baseline features only:** AUC = 0.590
* **Baseline + all query features:** AUC = 0.677
* **Baseline + safe query features:** AUC = 0.619
* **Baseline + suspect query features:** AUC = 0.648
* **Final leakage-clean model:** AUC = 0.619

The suspect-feature experiment was retained as a diagnostic rather than used for the final model. The final model uses only the baseline signals plus the two query-level features judged appropriate for the leakage-clean feature set.

## 5. Evaluation

The evaluation uses a **client-level holdout split**. Training contains **68,798 rows**, while the held-out test set contains **24,603 rows from 11 clients**.

The split is important because content from the same client can share structural patterns. Keeping test clients separate provides a more demanding evaluation than randomly mixing content items from the same clients between training and testing.

The test base rate is **73.9%**. This is reported alongside Precision@K because precision values should be interpreted relative to the underlying prevalence of the positive label.

The final model achieved:

* **AUC = 0.619**
* **Precision@20 = 0.950**
* **Precision@50 = 0.940**
* **Precision@100 = 0.920**
* **Precision@200 = 0.910**

The baseline achieved higher Precision@K at every reported cutoff:

* **Precision@20 = 1.000**
* **Precision@50 = 1.000**
* **Precision@100 = 0.950**
* **Precision@200 = 0.915**

The model therefore does not replace the baseline on top-k precision in this experiment.

The more important error-analysis finding is the disagreement between the two approaches. Using a 0.5 model-score threshold, **62,326 of 93,401 items (66.7%)** received different binary flags from the rule and model.

The rule flags **18,231 items (19.5%)**, while the model flags **76,011 items (81.4%)** at that threshold. Because the model score threshold was used only for diagnostic comparison, these numbers should not be interpreted as a validated production decision threshold.

## 6. Interpretation

The model found that query-level behavior adds directional information beyond the basic refresh signals.

Permutation importance on the held-out test data ranked the features approximately as follows:

| Feature                      | Permutation importance |
| ---------------------------- | ---------------------: |
| `impressions_prev30`         |                0.04895 |
| `query_concentration_prev30` |                0.04830 |
| `days_since_last_update`     |                0.01182 |
| `query_impressions_prev30`   |                0.00917 |
| `avg_position_prev30`        |                0.00019 |

The strongest measured signals were therefore **recent impressions and query concentration**, with freshness contributing additional signal. Average position contributed very little under this particular permutation-importance analysis.

The partial-dependence analysis for `impressions_prev30` showed that predicted declining probability generally increased from very low impression levels into the lower-thousands range, then flattened. For example, predicted probability was approximately **0.515 at 118 impressions**, **0.696 at 545**, and around **0.75–0.77 through much of the 1,400–6,500 impression range**.

This should be interpreted as an observed model relationship, not a causal effect of impressions on decline.

A major negative result is that the trained model does **not** outperform the transparent baseline on Precision@K. The useful finding is instead that the model produces a substantially different ranking. The top-500 overlap is only **1 item out of 500 (0.2%)**, suggesting that the model is identifying a different population of content rather than simply reproducing the baseline ordering.

The model top-500 has a much lower mean traffic level than the baseline top-500, approximately **2,470 versus 24,441 impressions/30d**. This supports the interpretation that the model is surfacing smaller pages that the visibility-based baseline largely ignores.

These are observed and directional relationships in this dataset. They do not establish causality or prove how a search engine ranks content.

## 7. Recommendation

The output should be used as a **ranked human-review playbook**, not an automated editing system.

### Priority 1 — Review high-scoring model items

Use the model score to identify content that may deserve review even when it does not satisfy the baseline rule. This is particularly useful for lower-traffic pages that would otherwise be absent from the baseline queue.

### Priority 2 — Review high-visibility baseline items

Continue using the transparent baseline for pages with substantial impressions that are stale or have weak observed position. These pages have strong practical visibility and should remain an important review population.

### Priority 3 — Investigate model/rule disagreements

The large disagreement set is useful for human review. Items where the model assigns a high score but the baseline assigns no action are especially useful for understanding what additional query-level patterns the model is capturing.

### Priority 4 — Use reason codes as review prompts

Reason codes should remain simple review explanations:

* `stale_but_visible` — high recent visibility and an update age of at least 180 days.
* `position_slipping` — average position is worse than the observed median.
* `standard_review` — does not satisfy either stronger rule condition.

These codes are **not causal diagnoses**. They explain the screening logic and should be verified by an editor.

### Confidence and limits

Confidence is moderate for using the output as a **screening and prioritization aid** within this dataset. Confidence is lower for claiming that the model will improve future traffic, rankings, or refresh outcomes.

The baseline currently has better measured Precision@K on the held-out test clients. The strongest practical case for the model is therefore not that it replaces the baseline, but that it provides a **different candidate pool**, especially among lower-traffic content.

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

The final model's measured AUC is **0.619**, while the baseline is stronger on the reported Precision@K metrics.

The model and baseline have only **1 overlapping item in their top-500 lists (0.2%)**, showing that they prioritize substantially different content.

No causal claims are made about content refreshes, traffic changes, or search-engine algorithms.

The analysis does not claim to predict Google's algorithm.

No client-identifying information is included.

The report's numerical claims are based on the completed capstone notebook results.
