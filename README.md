# KQYaili: OpenWebText 6-layer Transformer source snapshot

This branch preserves the existing local training source for a recorded **4.3614 validation loss**, evaluated at **43.0 minutes** on an **NVIDIA GeForce RTX 5070 Ti Laptop GPU**. Source files were copied byte-for-byte; no assignment implementation was added or rewritten for this upload. See SOURCE_SHA256SUMS.txt for the original-source hashes.

The recorded result is an unverified local run, not an official single-B200 reproduction. The 43.0-minute log timestamp is measured after final evaluation and before final plotting/checkpoint writing. A precise total process duration was not recorded.

## Recorded configuration

6-layer Pre-LN Transformer; hidden size 512; SwiGLU feed-forward size 1344; 16 attention heads; RoPE theta 10000; vocabulary size 32000; context length 512. Embedding weights are reset with trunc_normal(std=0.02, bounds=-0.06 to 0.06) and tied to the output head before optimizer construction.

AdamW; peak learning rate 1.2e-3; weight decay 0.01; gradient clipping 1.0; cosine decay to 1e-5; warmup 330 microbatches; 6600 microbatches total. Microbatch size 16 with 2-step gradient accumulation, yielding 3300 optimizer updates. The script selects TransformerLM; the unused UltimateTransformerLM class remains in the unmodified source.

## Existing data pipeline and entry points

The raw datasets are the course-provided OpenWebText training and validation files:

- https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
- https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

The local pipeline expects their decompressed files under data/. It also expects data/owt_train_sample.txt containing the first 134,217,728 bytes (128 MiB) of the training file. The sample's SHA256 is ffd13f60ce359e3914b8d57475707c62f2a2af2a8475ef1ee80e29ee7fad2ddd. The sample-generation step is not implemented in the uploaded source.

The existing cs336_basics/preprocess.py trains BPE from this sample, writes the tokenizer cache data/owt_bpe.pkl, and creates data/owt_train.bin and data/owt_valid.bin. An existing tokenizer cache is reused. The recorded training-token file contains 30,625,998 uint16 tokens. Preprocessing precedes the timed training run. Raw datasets, tokenizer caches, and checkpoints are not included in this branch.

train_owt.py is the training entry point and expects those token files relative to the project root. Each validation report averages 20 random batches of 16 sequences of length 512 (163,840 token positions), rather than traversing the full validation set.

## Reproduction status and limitations

The attached log was produced with the submitter's existing conda environment. A clean-environment rerun has not been performed. pyproject.toml and uv.lock are preserved as-is, but matplotlib, imported by the training script, is not declared in those dependencies. There is no main.py entry point; uv run main.py is not currently supported or claimed. Historical cache provenance and official B200 reproducibility have not been independently verified.

AI assistance was used for conceptual discussion, code review, and preparing the submission documentation. Eligibility under the official leaderboard's AI rule is not asserted. This branch supplies the source snapshot and evidence for maintainer review.

## Recorded evidence

- [Validation curve](evidence/leaderboard_curve.png)
- [Original training log](evidence/training.log)
- [Validation points](evidence/validation.csv)
