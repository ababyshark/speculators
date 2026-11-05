# GLM-5.3 (GlmMoeDsa) MTP=5 continued training on Atlas A3

Worked example: continued training (warm-start) of GLM-5.3-W4A8's native MTP layer (`model.layers.78`, DeepSeek-style layout with `eh_proj/enorm/hnorm/ shared_head`) with speculators, on 16 Ascend NPUs, then serving it back with vllm-ascend.

Read `docs/glm53-mtp-npu-training.md` first — it documents every problem, root cause, fix and measurement behind this example (MoE dispatch ~300x, large-vocab CE, FSDP granularity, DSA indexer zero-grad resume, W4A8 dequantization conventions, image/driver compatibility triage).

Scripts are hardcoded to the environment they were validated on (A3 node, `/mnt/weight2/l00840580-glm53-w4a8-biasfix`, CANN 9.1.0 container with vllm-ascend v0.28.0.rc1); adapt paths rather than assuming portability.

Pipeline:

1. `gen_onpolicy.py` — on-policy responses from the served W4A8 model (UltraChat prompts; any prompt source works).
2. `tokenize_onpolicy.py` — chat-template ids + assistant-span loss mask (speculator passthrough format).
3. `speculators prepare-data` — arrow dataset + token frequencies.
4. `serve_extract.sh` + `speculators generate-offline-data` — last-layer hidden states dumped to disk (`extract_hidden_states` spec mode + `ExampleHiddenStatesConnector`).
5. `extract_mtp_layer.py` — W4A8 layer-78 dequantization (BF16) + key remap into a speculators MTP warm-start checkpoint.
6. `train_t5.sh` — 16-rank FSDP continued training (MTP=5, fp32 masters).
7. Eval: `serve_eval_baseline.sh` (native MTP) vs `serve_eval_trained.sh` (`--spec-model` loads the speculators BF16 drafter directly), measured by `measure_acceptance.py` on 200 held-out prompts (`prep_eval_prompts.py`). `writeback_layer78.py` is the fallback serving path (BF16 layer-78 written back into a checkpoint copy).
