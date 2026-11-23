"""Account for the published V4.1 graph after DSpark semantic corrections."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


def draft_geometry(text_mask, supervision_mask, block, window):
    """Count supervised draft anchors and the main-context union they depend on.

    Masks describe one complete, unpadded document. Supervision is unshifted:
    anchor t conditions on x[t+1] and predicts x[t+2+j]. An active anchor keeps
    its entire noncausal draft block; anchors with no objective are excluded.
    """
    if len(text_mask) != len(supervision_mask) or block < 1 or window < 1:
        raise ValueError("Invalid draft mask or geometry")
    active = []
    targets = 0
    main_rows = set()
    for anchor in range(len(text_mask)):
        count = 0
        for offset in range(block):
            target = anchor + offset + 2
            if target >= len(text_mask) or not all(text_mask[anchor + 1:target + 1]):
                break
            count += bool(supervision_mask[target])
        if count:
            active.append(anchor)
            targets += count
            main_rows.update(range(max(0, anchor + 1 - window), anchor + 1))
    return {
        "active_anchors": len(active), "supervised_tokens": targets,
        "main_kv_tokens": len(main_rows), "block_size": block, "window": window,
        "attention_pairs": sum(block * (min(anchor + 1, window) + block) for anchor in active),
    }


def audit_geometry(metadata, data, max_seq_len, output):
    """Re-encode each distinct frozen image-caption record without alignment padding."""
    from hyper_parallel.data.constants import IGNORE_INDEX
    from hyper_parallel.models.deepseek_v41.adapter.data.processor import build_deepseek_v41_processor
    from hyper_parallel.models.deepseek_v41.adapter.data.transform_fn import build_deepseek_v41_omni_transform

    records = Counter(data.read_text(encoding="utf-8").splitlines())
    text_config = json.loads((metadata / "config.json").read_text(encoding="utf-8"))["text_config"]
    processor = build_deepseek_v41_processor(config_path=str(metadata))
    transform = build_deepseek_v41_omni_transform(
        processor=processor, training_mode="pretrain", sample_format="image_caption",
        max_seq_len=max_seq_len, sample_alignment_size=1,
    )
    geometry = []
    for line, count in records.items():
        sample = json.loads(line)
        sample["image"] = str((data.parent / sample["image"]).resolve())
        encoded = transform.encode_sample(sample)
        if encoded is None:
            raise ValueError("The frozen input contains an unencodable sample")
        valid = encoded["input_ids"].numel()
        if not max_seq_len // 2 < valid <= max_seq_len:
            raise ValueError("This accounting requires one complete image-caption record per microbatch")
        geometry.append({
            "records": count,
            "input_tokens": valid,
            "supervised_tokens": int((encoded["labels"] != IGNORE_INDEX).sum()),
            "engram_tokens": int((encoded["token_types"] < 0).sum()),
            "vision_patches": encoded["image_patch_lengths"].tolist(),
            "aligned_image_tokens": [h * w for h, w in encoded["image_llm_grid_hw"].tolist()],
            "dspark": draft_geometry(
                (encoded["token_types"] < 0).tolist(),
                (encoded["labels"] != IGNORE_INDEX).tolist(),
                text_config["dspark_block_size"], 128,
            ),
        })
    result = {"dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "records": geometry}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def required_flops(config, engram, sample, text_flops):
    """Use released widths and the logical sample geometry before padding and recompute."""
    text = config["text_config"]
    vision = config["vision_config"]
    seq = sample["input_tokens"]
    result = text_flops(text, seq)
    hidden, heads, dim = (text[key] for key in ("hidden_size", "num_attention_heads", "head_dim"))
    q_rank, o_rank = text["q_lora_rank"], text["o_lora_rank"]
    width, layers = vision["hidden_size"], vision["num_hidden_layers"]
    vision_weights = 4 * width**2 + 3 * width * vision["intermediate_size"]
    downsample = vision["downsample_ratio"]
    patches = sample["vision_patches"]
    image_tokens = sample["aligned_image_tokens"]
    if len(patches) != len(image_tokens) or not patches:
        raise ValueError("Every full-model sample must run the published vision tower")
    result["vision_patch_projection"] = 4 * sum(patches) * 3 * vision["patch_size"]**2 * width
    result["vision_blocks"] = layers * sum(6 * n * vision_weights + 12 * n**2 * width for n in patches)
    result["vision_aligner"] = 6 * sum(image_tokens) * (downsample**2 * width * hidden + hidden**2)
    hashes = (engram["max_ngram_size"] - 1) * engram["num_heads"]
    engram_tokens = sample["engram_tokens"]
    if not isinstance(engram_tokens, int) or not 0 <= engram_tokens <= seq:
        raise ValueError("Engram token count must match the non-image, non-padding input mask")
    # Image positions bypass Engram in the model. Computing their projection
    # before torch.where is implementation overhead, not required model work.
    result["engram_fusion_projection"] = (
        6 * engram_tokens * len(engram["layer_ids"]) * hashes * engram["head_dim"] * hidden * (text["hc_mult"] + 1)
    )

    draft_layers = text["num_nextn_predict_layers"]
    block = text["dspark_block_size"]
    draft = sample["dspark"]
    if draft["block_size"] != block or draft["window"] != 128:
        raise ValueError("Draft geometry differs from the published training contract")
    draft_rows = draft["active_anchors"] * block
    main_rows = draft["main_kv_tokens"]
    query_width = heads * dim
    draft_projection = hidden * q_rank + q_rank * query_width
    draft_projection += query_width * o_rank + text["o_groups"] * o_rank * hidden
    # Main and draft states share WKV. Every active anchor has a causal main
    # window and an additional noncausal K-by-K draft attention block.
    draft_projection += hidden * dim
    draft_pairs = draft["attention_pairs"]
    result["dspark_attention"] = draft_layers * (
        6 * (draft_rows * draft_projection + main_rows * hidden * dim) + 12 * query_width * draft_pairs
    )
    draft_experts = text["dspark_num_experts_per_tok"] + text["n_shared_experts"]
    result["dspark_experts_and_router"] = 6 * draft_layers * draft_rows * hidden * (
        3 * text["moe_intermediate_size"] * draft_experts + text["dspark_n_routed_experts"]
    )
    hc = text["hc_mult"]
    result["dspark_mhc_and_input_projection"] = (
        4 * main_rows * len(text["dspark_target_layer_ids"]) * hidden**2
        + 12 * draft_layers * draft_rows * hidden * hc * (2 * hc + hc**2)
    )
    markov_rank = text["dspark_markov_rank"]
    result["dspark_training_heads"] = draft["supervised_tokens"] * (
        4 * hidden * text["vocab_size"] + 6 * (markov_rank * text["vocab_size"] + hidden + markov_rank)
    )
    return result


def score(config, engram, geometry, log, batch, dies, first, last, output):
    """Score only complete, matching, unprofiled updates against 353 TFLOPS/die."""
    spec = importlib.util.spec_from_file_location("backbone_mfu", Path(__file__).with_name("measure-backbone-mfu.py"))
    backbone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backbone)
    records = geometry["records"]
    count = sum(row["records"] for row in records)
    totals = Counter()
    for row in records:
        components = required_flops(config, engram, row, backbone.model_flops)
        totals.update({key: value * row["records"] for key, value in components.items()})
    step_flops = {key: value * batch / count for key, value in totals.items()}
    step_input = sum(row["input_tokens"] * row["records"] for row in records) * batch / count
    step_targets = sum(row["supervised_tokens"] * row["records"] for row in records) * batch / count
    step_engram = sum(row["engram_tokens"] * row["records"] for row in records) * batch / count
    steps = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\| step=(\d+)\b", line)
        if match is None or "performance/step_time=" not in line:
            continue
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        step = int(match[1])
        seconds = float(fields["performance/step_time"])
        targets = float(fields["data/step_tokens"])
        if (
            step in steps or not math.isfinite(seconds) or seconds <= 0
            or float(fields["data/step_samples"]) != batch
            or targets != float(format(step_targets, ".6g"))
        ):
            raise ValueError(f"Step {step} does not match the declared geometry or unique timing record")
        steps[step] = seconds
    if set(steps) != set(range(1, last + 1)) or not 1 <= first <= last:
        raise ValueError(f"Expected complete updates 1 through {last}")
    seconds = sum(steps[index] for index in range(first, last + 1)) / (last - first + 1)
    report = {
        "method": "PaLM-style architecture-aware training GEMMs over non-padding inputs",
        "accounting_version": 3,
        "accounting_scope": {
            "engram": "WKV projection forward, input gradient and weight gradient on eligible text tokens",
            "elementwise_operations": "Not enumerated in this GEMM-dominant PaLM-style estimate",
            "lookup_and_transfers": "No FLOPs assigned; their time remains in the measured training step",
            "recompute_and_optimizer": "Excluded from MFU; reported separately in HFU",
            "dspark": "Main plus noncausal draft KV; active anchors only; supervised output heads; frozen backbone features have no input-gradient GEMM",
            "draft_boundary_approximation": "All K draft rows are charged in each active block, including partially supervised final blocks; this is a GEMM-dominant estimate, not an exact operation count",
        },
        "dataset_sha256": geometry["dataset_sha256"],
        "global_batch_size": batch, "dies": dies, "peak_tflops_per_die": 353,
        "timed_steps": [first, last], "mean_step_seconds": seconds,
        "input_tokens_per_step": step_input, "supervised_tokens_per_step": step_targets,
        "engram_eligible_tokens_per_step": step_engram,
        "supervised_tokens_per_second_per_die": step_targets / (seconds * dies),
        "input_tokens_per_second_per_die": step_input / (seconds * dies),
        "flops_per_step_by_component": step_flops,
        "mfu_percent": 100 * sum(step_flops.values()) / (353e12 * dies * seconds),
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    geometry = modes.add_parser("geometry")
    for name in ("metadata", "data", "output"):
        geometry.add_argument(name, type=Path)
    geometry.add_argument("--seq-len", required=True, type=int)
    measure = modes.add_parser("score")
    for name in ("config", "engram", "geometry", "log", "output"):
        measure.add_argument(name, type=Path)
    for name in ("batch", "dies", "first", "last"):
        measure.add_argument("--" + name, required=True, type=int)
    args = parser.parse_args()
    if args.mode == "geometry":
        audit_geometry(args.metadata, args.data, args.seq_len, args.output)
    else:
        inputs = [json.loads(path.read_text(encoding="utf-8")) for path in (args.config, args.engram, args.geometry)]
        score(*inputs, args.log, args.batch, args.dies, args.first, args.last, args.output)


if __name__ == "__main__":
    main()
