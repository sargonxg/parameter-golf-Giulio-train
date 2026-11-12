> **Portfolio maintenance status · 12 November 2026:** historical/reference scope; no further independent product expansion is planned. [Upstream project](https://github.com/openai/parameter-golf). [Purpose, retained work, retirement gates and restoration](TACITUS_RETIREMENT.md). Repository archival is a separate checked operation; this notice alone does not claim it occurred.

<!-- tacitus-fork-status -->
> **Retained upstream research fork.** The original project is [openai/parameter-golf](https://github.com/openai/parameter-golf).
> This fork is preserved for inspection and evaluation. Benchmark or performance
> claims in the upstream README below are upstream results, not new measurements
> of this fork. No model evaluation or hosted acceptance was run in this review.
> Original authorship and license terms remain in place.
> Training requires separately provisioned compute; no training job was run in this review.
<!-- /tacitus-fork-status -->

# Parameter Golf — historical experiment

<details>
<summary>TACITUS evaluation scope and optional original 8-bit teaching companion</summary>

<!-- tacitus-practical-entry -->
## Purpose, code and a playable introduction

Study reproducible model-training constraints before deciding whether a smaller model is useful for a bounded TACITUS extraction task.

**Upstream evaluation fork.** Upstream Parameter Golf experiment. No new TACITUS model, training run or independent benchmark result is claimed.

[Setup, code walkthrough, research & verification](TACITUS_GUIDE.md) · [Play the original 8-bit teaching adventure](tacitus-demo/index.html) · [Demo credits](tacitus-demo/CREDITS.md)

![The original House of Evidence teaching adventure; synthetic case, separate from this project’s runtime](tacitus-demo/mansion.png)

Research: [RAG](https://arxiv.org/abs/2005.11401), [long-context evidence use](https://arxiv.org/abs/2307.03172), [GraphRAG](https://arxiv.org/abs/2404.16130). [What TACITUS borrows, adds and still needs to measure](https://github.com/sargonxg/sargonxg/blob/main/docs/RESEARCH_AND_DESIGN.md).
<!-- /tacitus-practical-entry -->

</details>


> **Status: completed and no longer maintained.** This repository is retained as a historical fork and experiment record. It is not an active TACITUS project and should not be read as a current product or research release.

This repository was forked from [OpenAI's Parameter Golf challenge](https://github.com/openai/parameter-golf) in March 2026 to explore small-model training and conflict-domain evaluation ideas. The upstream project remains the authoritative source for the challenge, rules, leaderboard, and current implementation.

## What is preserved here

- the upstream challenge code at the time of the experiment;
- exploratory conflict-domain evaluation and predictor files;
- local training and model-compression experiments;
- the commit history needed to reproduce or inspect that work.

No maintained package, supported model, benchmark result, or TACITUS production component is published from this fork. For current TACITUS work, see [DIALECTICA](https://github.com/sargonxg/A3_DIALECTICAbyTACITUS_v3), [TCGC](https://github.com/sargonxg/TCGC_TACITUS-Conflict-Grammar-Corpus_BENCHMARK), and the [TACITUS profile](https://github.com/sargonxg).

## Attribution and license

The underlying challenge is the work of OpenAI and its contributors. See the repository history, [LICENSE](LICENSE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the applicable terms and attribution.
