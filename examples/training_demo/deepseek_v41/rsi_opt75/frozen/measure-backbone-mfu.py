"""Compute PaLM MFU for the frozen 552B V4.1 text-backbone series."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


SEQUENCE_LENGTH = 4096
DIES = 128
PEAK_PER_DIE = 353e12
PEAK = DIES * PEAK_PER_DIE
PA_LM_REFERENCE = "https://arxiv.org/pdf/2204.02311#page=66"
STEP_MARKER = re.compile(r"\bstep=(\d+)\s+epoch=\d+\b")


def model_flops(config, sequence_length=SEQUENCE_LENGTH, *, full_range_indexer=False):
    """Count model GEMMs on a full unpadded document, independent of parallel layout."""
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    experts = config["n_routed_experts"]
    topk = config["num_experts_per_tok"]
    if (layers, hidden, heads, head_dim, experts, topk) != (40, 5120, 64, 512, 384, 6):
        raise ValueError("Expected the released 552B text architecture")
    ratios = config["compress_ratios"][:layers]
    kv_sources = config["kv_source_layer_ids"]
    index_sources = config["index_source_layer_ids"]
    if len(ratios) != layers or not set(ratios) <= {0, 1, 2}:
        raise ValueError("Expected the released SWA and CED layer schedule")
    length = sequence_length
    query_width = heads * head_dim
    q_rank, o_rank = config["q_lora_rank"], config["o_lora_rank"]
    index_heads, index_dim = config["index_n_heads"], config["index_head_dim"]
    index_width = index_heads * index_dim
    index_topk = config["index_topk"]
    hc_streams = config["hc_mult"]
    positions = range(1, length + 1)

    attention_weights = hidden * (q_rank + head_dim) + q_rank * query_width
    attention_weights += query_width * o_rank + config["o_groups"] * o_rank * hidden
    expert_weights = 3 * hidden * config["moe_intermediate_size"]
    mixing_weights = 2 * hidden * hc_streams * (2 * hc_streams + hc_streams**2)
    window_pairs = sum(min(position, config["sliding_window"]) for position in positions)
    compressed_pairs = {ratio: sum(min(position // ratio, index_topk) for position in positions)
                        for ratio in (1, 2)}
    block_size = config["candidate_block_size"]
    candidate_capacity = block_size * config["candidate_topk_blocks"]
    candidate_source = config["candidate_source_layer_id"]
    if not full_range_indexer and candidate_capacity < index_topk:
        raise ValueError("The published candidate pool must cover the Indexer TopK width")

    def indexer_key_pairs(layer):
        causal_keys = (position // ratios[layer] for position in positions)
        if full_range_indexer or layer <= candidate_source:
            return sum(causal_keys)
        # Reindex retains the latest block even when it is only partly causal.
        return sum(keys if keys <= candidate_capacity
                   else candidate_capacity - block_size + (keys - 1) % block_size + 1
                   for keys in causal_keys)

    index_pairs = sum(indexer_key_pairs(layer) for layer in index_sources)
    index_grad_pairs = sum(compressed_pairs[ratios[layer]] for layer in index_sources)
    compressor_projections = sum(1 + int(ratios[layer] > 1) for layer in kv_sources)
    index_keys = sum(length // ratios[layer] for layer in kv_sources)

    # Indexer inputs and the KL teacher are detached. Charge a weight gradient
    # for the trainable indexer projections, and score derivatives on the
    # selected keys. Repeated evaluation of a logical score is implementation work.
    return {
        "attention_projections": 6 * length * layers * attention_weights,
        "shared_and_selected_expert_projections": (
            6 * length * layers * expert_weights * (topk + config["n_shared_experts"])
        ),
        "router_projection": 6 * length * layers * hidden * experts,
        "mhc_mixer_projections": 6 * length * layers * mixing_weights,
        "language_model_head": 6 * length * hidden * config["vocab_size"],
        "compressor_projections": 6 * length * compressor_projections * hidden * head_dim,
        "indexer_projections": (
            4 * length * len(index_sources) * (q_rank * index_width + hidden * index_heads)
            + 4 * index_keys * head_dim * index_dim
        ),
        "attention_qk_and_pv": (
            12 * query_width * (layers * window_pairs + sum(compressed_pairs.get(ratio, 0) for ratio in ratios))
        ),
        "indexer_qk_and_selected_gradients": index_width * (2 * index_pairs + 4 * index_grad_pairs),
    }


def read_steps(log, first, last, expected_tokens):
    """Read HP's cross-rank maximum wall time and global tokens for every timed step."""
    steps = {}
    with log.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            if "data/step_tokens=" not in line or "performance/step_time=" not in line:
                continue
            marker = STEP_MARKER.search(line)
            if marker is None or not 1 <= int(marker[1]) <= last:
                continue
            step = int(marker[1])
            fields = dict(field.split("=", 1) for field in line.split() if "=" in field)
            token_text = fields["data/step_tokens"]
            tokens = float(token_text)
            wall_time = float(fields["performance/step_time"])
            loss = float(fields["training/total_loss"])
            grad_norm = float(fields["training/grad_norm"])
            # HP renders scalar metrics with .6g. The frozen data and batch give the exact count.
            tokens_match = tokens == expected_tokens or token_text == format(float(expected_tokens), ".6g")
            if not tokens_match or not math.isfinite(wall_time) or wall_time <= 0 or step in steps:
                raise ValueError(f"Unexpected timing or token evidence for step {step}")
            if not math.isfinite(loss) or not math.isfinite(grad_norm) or grad_norm < 0:
                raise ValueError(f"Invalid loss or gradient norm at step {step}")
            steps[step] = {"step": step, "global_tokens": expected_tokens, "seconds": wall_time}
            if tokens != expected_tokens:
                steps[step]["logged_global_tokens"] = token_text
    if set(steps) != set(range(1, last + 1)):
        raise ValueError(f"Incomplete training window 1–{last}; found {sorted(steps)}")
    return [steps[step] for step in range(first, last + 1)]


def main():
    """Write component counts and optionally score one complete benchmark window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--full-range-indexer", action="store_true")
    parser.add_argument("--first-step", type=int, default=6)
    parser.add_argument("--last-step", type=int, default=15)
    args = parser.parse_args()
    if min(args.global_batch_size, args.sequence_length) < 1 or not 1 <= args.first_step <= args.last_step:
        parser.error("Use a positive batch size and sequence length, with a nonempty optimizer-step interval")
    raw = args.config.read_bytes()
    components = model_flops(json.loads(raw)["text_config"], args.sequence_length,
                             full_range_indexer=args.full_range_indexer)
    flops = sum(components.values())
    report = {
        "method": "PaLM-style required training GEMMs for DeepSeek-V4.1-Flash 552B text backbone",
        "reference": PA_LM_REFERENCE,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "sequence_length": args.sequence_length,
        "full_range_indexer": args.full_range_indexer,
        "global_batch_size": args.global_batch_size,
        "dies": DIES,
        "peak_flops_per_die": PEAK_PER_DIE,
        "peak_flops": PEAK,
        "flops_per_document_by_component": components,
        "flops_per_token": flops / args.sequence_length,
        "global_tokens_per_second_at_20_percent": 0.20 * PEAK * args.sequence_length / flops,
        "global_tokens_per_second_at_25_percent": 0.25 * PEAK * args.sequence_length / flops,
    }
    if args.log:
        steps = read_steps(args.log, args.first_step, args.last_step, args.global_batch_size * args.sequence_length)
        tokens = sum(step["global_tokens"] for step in steps)
        seconds = sum(step["seconds"] for step in steps)
        report.update(steps=steps, global_tokens_per_second=tokens / seconds,
                      tokens_per_second_per_die=tokens / (seconds * DIES),
                      mfu_percent=100 * tokens * report["flops_per_token"] / (seconds * PEAK))
        print(f'PALM_MFU_PERCENT={report["mfu_percent"]:.8f}')
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
