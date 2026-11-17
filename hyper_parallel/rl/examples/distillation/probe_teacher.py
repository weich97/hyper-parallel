# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Check a configured real teacher before starting student training."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.distillation.config import parse_distillation_config
from rl.distillation.teacher import TeacherManager


def main() -> None:
    """Validate teacher identity, tokenizer and echoed original token probabilities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-forward", action="store_true", help="Compare with an independent CPU HF forward")
    parser.add_argument("--decode-reference", action="store_true", help="Compare echo scores with prefix decode scores")
    parser.add_argument("--reference-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.005)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("Reference probability tolerances must be nonnegative")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    distillation = parse_distillation_config(config.get("distillation"))
    if not distillation.enabled:
        raise ValueError("Probe requires enabled distillation")
    manager = TeacherManager(distillation)
    manager.validate(config["model"]["tokenizer_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer_path"], local_files_only=True)
    tokens = tokenizer.encode("What is 2 + 2? The answer is 4.", add_special_tokens=False)
    for key, client in manager.clients.items():
        scores = client.score(tokens)
        if args.decode_reference:
            decoded = []
            for position in range(1, len(tokens)):
                # Public score() exercises echo; this reference must invoke the separate decode endpoint path.
                response = client._request("v1/completions", {  # pylint: disable=protected-access
                    "model": client.config.served_model_name, "prompt": tokens[:position],
                    "max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                    "allowed_token_ids": [tokens[position]], "logprobs": 0,
                    "return_tokens_as_token_ids": True, "return_token_ids": True,
                })
                logprobs = response["choices"][0]["logprobs"]
                if logprobs["tokens"] != [f"token_id:{tokens[position]}"]:
                    raise ValueError("Decode reference did not return the requested action token")
                decoded.append(logprobs["token_logprobs"][0])
            actual = torch.tensor(scores, dtype=torch.float32)
            expected = torch.tensor(decoded, dtype=torch.float32)
            difference = (actual - expected).abs()
            print(json.dumps({"teacher": key, "reference": "independent-prefix-decode",
                              "positions": len(scores), "max_abs_difference": float(difference.max()),
                              "mean_abs_difference": float(difference.mean()),
                              "atol": args.atol, "rtol": args.rtol,
                              "echo_scores": scores, "decode_scores": decoded}), flush=True)
            torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
        if args.reference_forward:
            torch.set_num_threads(8)
            model = AutoModelForCausalLM.from_pretrained(
                client.config.model_path, local_files_only=True, dtype=getattr(torch, args.reference_dtype),
                attn_implementation="eager",
            ).eval()
            input_ids = torch.tensor([tokens], dtype=torch.long)
            with torch.inference_mode():
                logits = model(input_ids=input_ids, use_cache=False).logits[0, :-1].float()
                expected = logits.log_softmax(-1).gather(-1, input_ids[0, 1:, None]).squeeze(-1)
            actual = torch.tensor(scores, dtype=torch.float32)
            difference = (actual - expected).abs()
            evidence = {"teacher": key, "positions": len(scores), "reference_dtype": args.reference_dtype,
                        "max_abs_logprob_difference": float(difference.max()),
                        "mean_abs_logprob_difference": float(difference.mean()),
                        "atol": args.atol, "rtol": args.rtol,
                        "api_scores": scores, "reference_scores": expected.tolist()}
            print(json.dumps(evidence), flush=True)
            torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
            del model
        print(f"{key}: scored {len(scores)} next-token positions; min={min(scores):.6f}, max={max(scores):.6f}")


if __name__ == "__main__":
    main()
