# Non-record WIP: OAG Neurosymbolic Hybrid

## Approach
Ontology-Augmented Generation applied to text compression. Three predictors matched to web text entropy:
- **Neural transformer** (SOTA stack) — handles semantic uncertainty
- **FST grammar** (67 tags, 73 boilerplate, 17 phrases) — deterministic structural prediction
- **N-gram cache** (order-5, backward-looking) — adaptive local patterns, 95.8% coverage
- **Entropy-adaptive gating** — trusts cache more when neural is uncertain

## Status
Work in progress. Preliminary results on 1xH100. Requesting 8xH100 for full evaluation.

## Files
- `train_gpt.py` — baseline training
- `fst_predictor.py` — FST structural predictor (zero artifact cost)
- `ngram_cache.py` — backward-looking n-gram cache (confirmed legal Mar 25)
- `hybrid_eval.py` — unified hybrid evaluation with entropy-adaptive mixing
