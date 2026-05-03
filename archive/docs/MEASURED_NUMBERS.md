# Measured Numbers Log — All Datasets

**Extraction Date:** May 3, 2026  
**Status:** COMPLETE (all previously pending placeholders filled)

## NQ

| Field | Value |
|---|---|
| raw_source_count | 6,515 |
| usable_count_after_filtering | 6,515 |
| gold_ingestion_unit_count | 51,251 |
| base_distractor_doc_count | 82,326 |
| base_distractor_chunk_count | 50,000 |
| eligible_nonmember_count | NA |
| field_mapping | NA |
| notes | Source split: dpr-w100/natural-questions/dev. Query-doc qrels: 979,893. |

## TriviaQA

| Field | Value |
|---|---|
| raw_source_count | 17,944 |
| usable_count_after_filtering | 16,137 |
| gold_ingestion_unit_count | 29,167 |
| base_distractor_doc_count | 6,407,814 |
| base_distractor_chunk_count | 50,000 |
| eligible_nonmember_count | 27,380 |
| field_mapping | NA |
| notes | eligible_nonmember_count is gold-unit candidates after excluding ingested gold units (ingestion_size=1000, seed=42). Pair-level non-member pool: 15,137. |

## PubMedQA

| Field | Value |
|---|---|
| raw_source_count | 1,000 |
| usable_count_after_filtering | 1,000 |
| gold_ingestion_unit_count | 3,358 |
| base_distractor_doc_count | 27,738,441 |
| base_distractor_chunk_count | 50,000 |
| eligible_nonmember_count | 0 |
| field_mapping | NA |
| notes | With ingestion_size=1000, all usable items are ingested, leaving no eligible non-member gold units. |

## FinDER

| Field | Value |
|---|---|
| raw_source_count | 5,703 |
| usable_count_after_filtering | 5,696 |
| gold_ingestion_unit_count | 6,113 |
| base_distractor_doc_count | 8,055,455 |
| base_distractor_chunk_count | 50,000 |
| effective_pool_after_holdouts | 3,896 |
| field_mapping | question=text; evidence=references; answer=answer |
| notes | effective_pool_after_holdouts computed as 5,696 - 1,800 holdouts, where holdouts are 1000 ingestion + 200 benign + 100 poisoning + 400 MIA + 100 leakage. |

## Verification Notes

- Base distractor chunk counts were verified directly from local Chroma base DB collections and are all 50,000.
- Distractor doc counts were taken from Hugging Face dataset split metadata (`train` split) to avoid full re-download.
- Non-member and holdout counts are computed using project defaults in [config/default_utility_config.yaml](config/default_utility_config.yaml): ingestion_size=1000, ingestion_seed=42.

